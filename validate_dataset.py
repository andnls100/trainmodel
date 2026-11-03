#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文短剧：剧本格式 + SFT 数据集校验器。

依赖：仅标准库（schema 校验为内置的 Draft-07 子集实现）；可选 transformers（统计 token 长度）。

用法
  # 1) 校验一份完整剧本（含 场次总表 / 台词本，重新计算并对账）
  python validate_dataset.py script novel-script.md

  # 2) 校验 JSONL 数据集；同时传入多个划分文件，才能检查作品泄漏
  python validate_dataset.py data sft_data/train.jsonl sft_data/validation.jsonl sft_data/test.jsonl
  #    常用：--fix-est 回填预估秒数  --export out_dir 导出训练用 jsonl  --tokenizer Qwen/Qwen3-4B-Instruct-2507

ERROR = 必须修；WARN = 建议看一眼。有 ERROR 时退出码为 1。
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

# ---------------- 格式常量（来自示例剧本，已用其场次总表/台词本反算核对） ----------------
ACTION_SEC = 2.5          # 每个动作行 2.5 秒
CHARS_PER_SEC = 4.5       # 台词：字符数（含标点）/ 4.5 秒
NARRATOR = "画外音"        # 不需要出现在『人物』里的特殊说话人
BAD_TOKENS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")
ROUND1_TASKS = {"episode_generation", "scene_continuation", "scene_rewrite", "instruction_edit"}
CFG = {"sec_tol": 0.15, "hook_max": 15.0}

RE_TITLE = re.compile(r"^# (.+?) · 剧本（第 (\d+)[–-](\d+) 集）$")
RE_EP = re.compile(r"^## 第 (\d+) 集$")
RE_META1 = re.compile(r"^> 预估 ([\d.]+) 秒 / 目标 (\d+) 秒 · 开场钩子：(.+?)（第 (\d+) 场第 (\d+) 拍兑现） · 结尾悬念：(.+)$")
RE_META2 = re.compile(r"^> 认领爽点：(.+)$")
RE_SCENE = re.compile(r"^### 第 (\d+) 场 · (.+?)（([^（）]+)）$")
RE_CAST = re.compile(r"^人物：([^　]+?)(?:　道具：(.+))?$")   # 道具前是全角空格 U+3000
RE_DLG = re.compile(r"^\*\*([^*]+)\*\*（([^）]*)）：(.+)$")


class Report:
    def __init__(self):
        self.errors, self.warns = [], []

    def err(self, m):
        self.errors.append(m)

    def warn(self, m):
        self.warns.append(m)


# ---------------- 计时 ----------------
def beat_sec(b):
    return ACTION_SEC if b[0] == "action" else len(b[2]) / CHARS_PER_SEC


def scene_sec(sc):
    return round(sum(beat_sec(b) for b in sc["beats"]), 1)


def ep_sec(ep):
    return round(sum(scene_sec(s) for s in ep["scenes"]), 1)


def dlg_count(sc):
    return sum(b[0] == "dialogue" for b in sc["beats"])


# ---------------- 解析 ----------------
def parse_body(lines, rep, tag):
    """lines: [(行号, 文本)]。解析若干个『### 场』及其人物行 / 动作 / 台词，返回 scenes。
    节拍(beat) = 场内每一行动作或台词，按出现顺序编号（『第 K 场第 M 拍』指它）。"""
    scenes, cur, prev_body, crowded = [], None, False, 0
    for no, raw in lines:
        s = raw.strip()
        if not s:
            prev_body = False
            continue
        loc = f"{tag} L{no}"
        m = RE_SCENE.match(s)
        if m:
            cur = dict(num=int(m[1]), name=m[2], light=m[3], cast=None, props=[], beats=[], line=no)
            scenes.append(cur)
            prev_body = False
            continue
        if s.startswith("#"):
            rep.err(f"{loc} 标题行不合规（应为『### 第 N 场 · 场景名（光照）』）：{s[:40]}")
            continue
        if cur is None:
            rep.err(f"{loc} 第一个场景标题之前出现内容：{s[:40]}")
            continue
        if s.startswith((">", "|")) or s == "---":
            rep.err(f"{loc} 正文中不允许出现引用/表格/分隔线：{s[:40]}")
            continue
        mc = RE_CAST.match(s)
        if cur["cast"] is None:
            if mc:
                cur["cast"] = [x.strip() for x in mc[1].split("、") if x.strip()]
                cur["props"] = [x.strip() for x in (mc[2] or "").split("、") if x.strip()]
                prev_body = True
                continue
            rep.err(f"{loc} 场景标题后的第一行必须是『人物：甲、乙　道具：丙』")
            cur["cast"] = []
        elif mc:
            rep.err(f"{loc} 同一场景出现多个『人物：』行")
            continue
        if prev_body:
            crowded += 1
        prev_body = True
        md = RE_DLG.match(s)
        if md:
            who, tone, text = md[1].strip(), md[2].strip(), md[3].strip()
            if not tone:
                rep.err(f"{loc}『{who}』括号内的语气/动作为空")
            if who != NARRATOR and who not in cur["cast"]:
                rep.err(f"{loc} 说话人『{who}』不在本场『人物』中（旁白请用『{NARRATOR}』）")
            cur["beats"].append(("dialogue", who, text))
        elif s.startswith("**"):
            rep.err(f"{loc} 疑似台词但格式不合规（应为 **角色**（语气）：台词，括号与冒号为全角）：{s[:40]}")
        else:
            names = "|".join(re.escape(n) for n in cur["cast"] + [NARRATOR])
            if re.match(rf"^(?:{names})\s*(?:[（(][^）)]*[）)])?\s*[：:]", s):
                rep.err(f"{loc} 台词没有使用加粗格式，会被当成动作行：{s[:40]}")
            cur["beats"].append(("action", "", s))
    if crowded:
        rep.warn(f"{tag} 有 {crowded} 处相邻正文行之间没有空行（每个节拍独占一段、中间空一行）")
    for sc in scenes:
        if sc["cast"] is None:
            rep.err(f"{tag} 第 {sc['num']} 场缺少『人物：』行")
            sc["cast"] = []
        if not sc["beats"]:
            rep.err(f"{tag} 第 {sc['num']} 场没有任何动作或台词")
        if len(set(sc["cast"])) != len(sc["cast"]):
            rep.warn(f"{tag} 第 {sc['num']} 场『人物』有重复项")
    return scenes


def check_numbering(scenes, rep, tag, from_one):
    nums = [s["num"] for s in scenes]
    if not nums:
        rep.err(f"{tag} 没有任何『### 第 N 场』")
        return
    start = 1 if from_one else nums[0]
    if nums != list(range(start, start + len(nums))):
        rep.err(f"{tag} 场次编号应从 {start} 起连续递增，实际为 {nums}")


def parse_episode(lines, rep, tag):
    """lines: [(行号, 文本)]，以『## 第 N 集』开头。header 不合规返回 None。"""
    idx = [i for i, (_, l) in enumerate(lines) if l.strip()][:3]
    if len(idx) < 3:
        rep.err(f"{tag} 内容过短，缺少集标题 / 概要行")
        return None
    h = [lines[i][1].strip() for i in idx]
    m0, m1, m2 = RE_EP.match(h[0]), RE_META1.match(h[1]), RE_META2.match(h[2])
    for m, name, line in ((m0, "集标题『## 第 N 集』", h[0]), (m1, "概要行『> 预估 X 秒 / 目标 Y 秒 · 开场钩子：…（第 K 场第 M 拍兑现） · 结尾悬念：…』", h[1]),
                          (m2, "爽点行『> 认领爽点：甲、乙』", h[2])):
        if not m:
            rep.err(f"{tag} {name}格式不符：{line[:50]}")
    if not (m0 and m1 and m2):
        return None
    ep = dict(num=int(m0[1]), est=float(m1[1]), target=int(m1[2]), hook=m1[3], hook_scene=int(m1[4]),
              hook_beat=int(m1[5]), ending=m1[6], tags=[t.strip() for t in m2[1].split("、") if t.strip()])
    ep["scenes"] = parse_body(lines[idx[2] + 1:], rep, tag)
    check_numbering(ep["scenes"], rep, tag, True)
    if not ep["tags"]:
        rep.warn(f"{tag} 『认领爽点』为空")
    real = ep_sec(ep)
    if abs(real - ep["est"]) > 0.2:
        rep.err(f"{tag}『预估 {ep['est']} 秒』与正文实测 {real} 秒不符（--fix-est 可回填）")
    if ep["target"] > 0 and abs(real - ep["target"]) > ep["target"] * CFG["sec_tol"]:
        rep.err(f"{tag} 实测 {real} 秒，偏离目标 {ep['target']} 秒超过 ±{CFG['sec_tol']:.0%}")
    sc = {s["num"]: s for s in ep["scenes"]}.get(ep["hook_scene"])
    if sc is None or not 1 <= ep["hook_beat"] <= len(sc["beats"]):
        rep.err(f"{tag} 钩子兑现位置『第 {ep['hook_scene']} 场第 {ep['hook_beat']} 拍』不存在")
    else:
        before = sum(scene_sec(s) for s in ep["scenes"] if s["num"] < ep["hook_scene"]) \
            + sum(beat_sec(b) for b in sc["beats"][:ep["hook_beat"] - 1])
        if before > CFG["hook_max"]:
            rep.warn(f"{tag} 开场钩子约在 {before:.1f} 秒才兑现（建议 ≤ {CFG['hook_max']:.0f} 秒）")
    return ep


def fix_est(text):
    """按正文实测值回填『预估 X 秒』。模型数不准秒数，建议由脚本回填。"""
    ep = parse_episode(list(enumerate(text.split("\n"), 1)), Report(), "fix")
    if not ep or not ep["scenes"]:
        return text
    return re.sub(r"(> 预估 )[\d.]+( 秒 /)", lambda m: f"{m[1]}{ep_sec(ep):.1f}{m[2]}", text, count=1)


# ---------------- 完整剧本（含两张总表） ----------------
def md_table(lines, title):
    texts = [l.strip() for _, l in lines]
    if title not in texts:
        return None
    i = texts.index(title) + 1
    while i < len(texts) and not texts[i].startswith("|"):
        if texts[i].startswith("## "):
            return None
        i += 1
    rows = []
    while i < len(texts) and texts[i].startswith("|"):
        rows.append([c.strip() for c in texts[i].strip("|").split("|")])
        i += 1
    return rows[2:]   # 去掉表头和分隔行


def check_script_doc(path):
    rep = Report()
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    for t in BAD_TOKENS:
        if t in text:
            rep.err(f"文件含特殊 token 文本 {t}")
    lines = list(enumerate(text.split("\n"), 1))
    cut = next((i for i, (_, l) in enumerate(lines) if l.strip() == "## 场次总表"), None)
    body = lines[:cut] if cut is not None else lines
    while body and body[-1][1].strip() in ("", "---"):
        body.pop()
    first = next((l.strip() for _, l in body if l.strip()), "")
    mt = RE_TITLE.match(first)
    if not mt:
        rep.err(f"首行应为『# 作品名 · 剧本（第 a–b 集）』，实际：{first[:40]}")
    starts = [i for i, (_, l) in enumerate(body) if RE_EP.match(l.strip())]
    eps = []
    for k, st in enumerate(starts):
        en = starts[k + 1] if k + 1 < len(starts) else len(body)
        ep = parse_episode(body[st:en], rep, f"第{RE_EP.match(body[st][1].strip())[1]}集")
        if ep:
            eps.append(ep)
    nums = [e["num"] for e in eps]
    if not starts:
        rep.err("没有找到任何『## 第 N 集』")
    elif nums != list(range(nums[0], nums[0] + len(nums))) or len(nums) != len(starts):
        rep.err(f"集编号应连续递增，实际为 {nums}")
    if mt and nums and (int(mt[2]), int(mt[3])) != (nums[0], nums[-1]):
        rep.err(f"标题写的是第 {mt[2]}–{mt[3]} 集，正文是第 {nums[0]}–{nums[-1]} 集")
    if cut is None:
        rep.err("缺少『## 场次总表』")
        return rep, eps
    check_tables(lines, eps, rep)
    return rep, eps


def check_tables(lines, eps, rep):
    exp = [(e["num"], s["num"], s["name"], s["light"], "、".join(s["cast"]), dlg_count(s), scene_sec(s))
           for e in eps for s in e["scenes"]]
    t1 = md_table(lines, "## 场次总表")
    if t1 is None:
        rep.err("缺少『## 场次总表』表格")
    elif len(t1) != len(exp):
        rep.err(f"场次总表有 {len(t1)} 行，实际场次 {len(exp)} 个")
    else:
        for r, x in zip(t1, exp):
            try:
                ok = (int(r[0]), int(r[1]), r[2], r[3], r[4]) == x[:5] and int(r[5]) == x[5] and abs(float(r[6]) - x[6]) <= 0.15
            except (ValueError, IndexError):
                ok = False
            if not ok:
                rep.err(f"场次总表 第{x[0]}集第{x[1]}场 与正文不符：表={r}，实测={list(x)}")
    agg = collections.defaultdict(lambda: [0, 0])
    for e in eps:
        for s in e["scenes"]:
            for kind, who, txt in s["beats"]:
                if kind == "dialogue":
                    agg[who][0] += 1
                    agg[who][1] += len(txt)
    t2 = md_table(lines, "## 台词本")
    if t2 is None:
        rep.err("缺少『## 台词本』表格")
        return
    try:
        got = {r[0]: (int(r[1]), int(r[2]), float(r[3])) for r in t2}
    except (ValueError, IndexError):
        rep.err("台词本表格无法解析（应为 角色 | 台词句数 | 字数 | 约合秒数）")
        return
    for w in sorted(set(got) | set(agg)):
        g, a = got.get(w), agg.get(w)
        x = (a[0], a[1], round(a[1] / CHARS_PER_SEC, 1)) if a else None
        if g is None or x is None or g[:2] != x[:2] or abs(g[2] - x[2]) > 0.15:
            rep.err(f"台词本『{w}』与正文不符：表={g}，实测={x}")


# ---------------- 内置的 JSON Schema（Draft-07 子集）校验 ----------------
_TYPES = {"string": lambda v: isinstance(v, str), "array": lambda v: isinstance(v, list),
          "object": lambda v: isinstance(v, dict), "null": lambda v: v is None,
          "boolean": lambda v: isinstance(v, bool),
          "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
          "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)}


def schema_errors(v, sch, root, path="", out=None):
    """支持：$ref type enum const required properties additionalProperties(false) items minItems maxItems
    uniqueItems minLength minimum maximum pattern allOf if/then/else。返回错误字符串列表。"""
    out = [] if out is None else out
    if "$ref" in sch:
        node = root
        for k in sch["$ref"].lstrip("#/").split("/"):
            node = node[k]
        return schema_errors(v, node, root, path, out)

    def bad(msg):
        out.append(f"{path or '<root>'}: {msg}")

    if "type" in sch:
        ts = sch["type"] if isinstance(sch["type"], list) else [sch["type"]]
        if not any(_TYPES[t](v) for t in ts):
            bad(f"类型应为 {ts}")
            return out
    if "const" in sch and v != sch["const"]:
        bad(f"应为 {sch['const']!r}，实际 {str(v)[:30]!r}")
    if "enum" in sch and v not in sch["enum"]:
        bad(f"应为 {sch['enum']} 之一，实际 {str(v)[:30]!r}")
    if isinstance(v, str):
        if len(v) < sch.get("minLength", 0):
            bad(f"长度不足 {sch['minLength']}")
        if "pattern" in sch and not re.search(sch["pattern"], v):
            bad(f"不符合格式 {sch['pattern']}（开头：{v[:20]!r}）")
    if _TYPES["number"](v):
        if v < sch.get("minimum", v):
            bad(f"不能小于 {sch['minimum']}")
        if v > sch.get("maximum", v):
            bad(f"不能大于 {sch['maximum']}")
    if isinstance(v, list):
        if len(v) < sch.get("minItems", 0):
            bad(f"至少 {sch['minItems']} 项")
        if len(v) > sch.get("maxItems", len(v)):
            bad(f"至多 {sch['maxItems']} 项")
        if sch.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in v}) != len(v):
            bad("有重复项")
        for i, x in enumerate(v):
            if "items" in sch:
                schema_errors(x, sch["items"], root, f"{path}/{i}", out)
    if isinstance(v, dict):
        for k in sch.get("required", []):
            if k not in v:
                bad(f"缺少字段 {k}")
        props = sch.get("properties", {})
        for k, x in v.items():
            if k in props:
                schema_errors(x, props[k], root, f"{path}/{k}", out)
            elif sch.get("additionalProperties") is False:
                bad(f"不允许的字段 {k}")
    for sub in sch.get("allOf", []):
        schema_errors(v, sub, root, path, out)
    if "if" in sch:
        branch = "then" if not schema_errors(v, sch["if"], root, path, []) else "else"
        if branch in sch:
            schema_errors(v, sch[branch], root, path, out)
    return out


# ---------------- 数据集 ----------------
def check_record(rec, schema, args, tag):
    rep = Report()
    for m in schema_errors(rec, schema, schema):
        rep.err(f"{tag} schema {m[:160]}")
    if rep.errors:
        return rep
    meta, prompt = rec["meta"], rec["prompt"]
    comp = rec["completion"][0]["content"].replace("\r\n", "\n")
    if prompt[-1]["role"] != "user":
        rep.err(f"{tag} prompt 最后一条必须是 user")
    if meta["task_type"] not in ROUND1_TASKS and not args.allow_round2:
        rep.err(f"{tag} task_type={meta['task_type']} 属第二轮任务（加 --allow-round2 放行）")
    for t in BAD_TOKENS:
        if t in comp or any(t in m["content"] for m in prompt):
            rep.err(f"{tag} 文本中含特殊 token 字面量 {t}（模板标记只能由 chat template 生成）")
    if args.fix_est and meta["output_kind"] == "episode":
        comp = fix_est(comp)
    rec["completion"][0]["content"] = comp
    lines = list(enumerate(comp.split("\n"), 1))
    if meta["output_kind"] == "episode":
        ep = parse_episode(lines, rep, tag)
        c = meta.get("constraints", {})
        if ep:
            if "episode_index" in c and c["episode_index"] != ep["num"]:
                rep.err(f"{tag} 要求第 {c['episode_index']} 集，正文是第 {ep['num']} 集")
            if "target_seconds" in c and c["target_seconds"] != ep["target"]:
                rep.err(f"{tag} 要求目标 {c['target_seconds']} 秒，正文写的是 {ep['target']} 秒")
            miss = [t for t in c.get("required_tags", []) if t not in ep["tags"]]
            if miss:
                rep.err(f"{tag} 缺少要求的爽点标签：{'、'.join(miss)}")
            rec["_stats"] = dict(sec=ep_sec(ep), scenes=len(ep["scenes"]))
    else:
        scenes = parse_body(lines, rep, tag)
        check_numbering(scenes, rep, tag, False)
    return rep


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def check_dataset(args):
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 无法加载 tokenizer {args.tokenizer}：{e}")
    glob = Report()
    recs, ids = [], collections.Counter()
    for p in args.files:
        for ln, line in enumerate(Path(p).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            tag = f"{Path(p).name}:{ln}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                glob.err(f"{tag} 不是合法 JSON：{e}")
                continue
            if isinstance(rec, dict) and isinstance(rec.get("meta"), dict) and "id" in rec["meta"]:
                tag += f" [{rec['meta']['id']}]"
                ids[rec["meta"]["id"]] += 1
            rep = check_record(rec, schema, args, tag)
            stem = Path(p).stem
            if not rep.errors and stem in ("train", "validation", "test") and rec["meta"]["split"] != stem:
                rep.err(f"{tag} meta.split={rec['meta']['split']} 与文件名 {stem} 不一致")
            n_tok = None
            if tok and not rep.errors:
                text = tok.apply_chat_template(rec["prompt"] + rec["completion"], tokenize=False)
                n_tok = len(tok(text, add_special_tokens=False).input_ids)
                if n_tok > args.max_len:
                    rep.warn(f"{tag} {n_tok} token，超过 max_len={args.max_len} 会被截断")
            recs.append((tag, rec, rep, n_tok))
    ok = [(t, r, n) for t, r, rep, n in recs if not rep.errors]
    # 全局：id、作品泄漏、衍生样本、重复
    for i, c in ids.items():
        if c > 1:
            glob.err(f"meta.id 重复：{i}（{c} 次）")
    by_id = {r["meta"]["id"]: r["meta"] for _, r, _ in ok}
    w2s = collections.defaultdict(set)
    for _, r, _ in ok:
        w2s[r["meta"]["work_id"]].add(r["meta"]["split"])
    for w, s in w2s.items():
        if len(s) > 1:
            glob.err(f"作品泄漏：work_id={w} 同时出现在 {sorted(s)}")
    for t, r, _ in ok:
        pid = r["meta"].get("parent_id")
        if pid and pid in by_id:
            pm = by_id[pid]
            if (pm["work_id"], pm["split"]) != (r["meta"]["work_id"], r["meta"]["split"]):
                glob.err(f"{t} 与来源样本 {pid} 的 work_id/split 不一致")
        elif pid:
            glob.warn(f"{t} parent_id={pid} 不在本次传入的文件里，无法核对")
        elif r["meta"]["task_type"] != "episode_generation":
            glob.warn(f"{t} 衍生任务建议填写 parent_id，便于检查泄漏")
    dup = collections.defaultdict(list)
    for t, r, _ in ok:
        dup[hashlib.md5(r["completion"][0]["content"].strip().encode()).hexdigest()].append(r["meta"])
    for ms in dup.values():
        if len(ms) > 1:
            names = ", ".join(m["id"] for m in ms[:4])
            (glob.err if len({m["split"] for m in ms}) > 1 else glob.warn)(f"completion 完全重复（{'跨划分' if len({m['split'] for m in ms}) > 1 else '同划分'}）：{names}")
    # 输出
    flat = [(rep.errors, rep.warns) for _, _, rep, _ in recs if rep.errors or rep.warns]
    for errs, warns in flat[:args.max_show]:
        for m in errs:
            print("ERROR", m)
        for m in warns:
            print("WARN ", m)
    if len(flat) > args.max_show:
        print(f"... 另有 {len(flat) - args.max_show} 条有问题的样本未显示（--max-show 调大）")
    for m in glob.errors:
        print("ERROR", m)
    for m in glob.warns:
        print("WARN ", m)
    n_err = sum(bool(rep.errors) for _, _, rep, _ in recs) + len(glob.errors)
    print(f"\n共 {len(recs)} 条，通过 {len(ok)} 条；有 ERROR 的样本 {sum(bool(rep.errors) for _, _, rep, _ in recs)} 条，全局 ERROR {len(glob.errors)} 条")
    if ok:
        print("划分：", dict(collections.Counter(r['meta']['split'] for _, r, _ in ok)),
              "任务：", dict(collections.Counter(r['meta']['task_type'] for _, r, _ in ok)))
        print("来源：", dict(collections.Counter(r['meta']['source'] for _, r, _ in ok)),
              "审核：", dict(collections.Counter(r['meta']['review_status'] for _, r, _ in ok)))
        secs = [r["_stats"]["sec"] for _, r, _ in ok if "_stats" in r]
        if secs:
            print(f"单集实测秒数：min {min(secs)} / p50 {pct(secs, .5)} / max {max(secs)}（{len(secs)} 集）")
        lens = [len(r["completion"][0]["content"]) for _, r, _ in ok]
        print(f"completion 字符数：p50 {pct(lens, .5)} / p90 {pct(lens, .9)} / max {max(lens)}")
        toks = [n for _, _, n in ok if n]
        if toks:
            over = sum(n > args.max_len for n in toks)
            print(f"token 长度：p50 {pct(toks, .5)} / p90 {pct(toks, .9)} / max {max(toks)}；超过 {args.max_len} 的 {over} 条（{over / len(toks):.1%}）")
    if args.export:
        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for _, r, _ in ok:
            sp = r["meta"]["split"]
            for suffix, obj in ((".jsonl", {"prompt": r["prompt"], "completion": r["completion"]}),
                                (".meta.jsonl", r["meta"])):
                files.setdefault(sp + suffix, []).append(json.dumps(obj, ensure_ascii=False))
        for name, rows in files.items():
            (out / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"已导出 {len(ok)} 条通过校验的样本到 {out}/（训练文件只含 prompt/completion，meta 单独存放）")
    return n_err


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sec-tol", type=float, default=0.15, help="实测秒数相对目标秒数的允许偏差，默认 0.15")
    common.add_argument("--hook-max-sec", type=float, default=15.0, help="开场钩子最晚兑现秒数（WARN），默认 15")
    common.add_argument("--max-show", type=int, default=40, help="最多显示多少条有问题的样本")
    ap = argparse.ArgumentParser(description="中文短剧剧本格式 / SFT 数据集校验")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("script", parents=[common], help="校验一份完整剧本 .md")
    s.add_argument("file")
    d = sub.add_parser("data", parents=[common], help="校验 JSONL 数据集")
    d.add_argument("files", nargs="+")
    d.add_argument("--schema", default=str(Path(__file__).with_name("dataset_schema.json")))
    d.add_argument("--allow-round2", action="store_true", help="放行第二轮任务类型")
    d.add_argument("--fix-est", action="store_true", help="按正文实测值回填『预估 X 秒』（配合 --export 落盘）")
    d.add_argument("--export", metavar="DIR", help="把通过校验的样本导出为训练用 jsonl（不含 meta）")
    d.add_argument("--tokenizer", help="如 Qwen/Qwen3-4B-Instruct-2507；给出才统计 token 长度")
    d.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()
    CFG["sec_tol"], CFG["hook_max"] = args.sec_tol, args.hook_max_sec
    if args.cmd == "script":
        rep, eps = check_script_doc(args.file)
        for m in rep.errors:
            print("ERROR", m)
        for m in rep.warns:
            print("WARN ", m)
        print(f"\n{len(eps)} 集，{sum(len(e['scenes']) for e in eps)} 场；ERROR {len(rep.errors)}，WARN {len(rep.warns)}")
        sys.exit(1 if rep.errors else 0)
    sys.exit(1 if check_dataset(args) else 0)


if __name__ == "__main__":
    main()
