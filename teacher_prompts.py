#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""教师模型提示词 + 数据生产胶水（与 validate_dataset.py、dataset_schema.json 放同一目录）。

本文件不调用任何 API：你自己把 messages 发给教师 / 裁判，再把返回文本交给对应函数。

流程
  0. render_story_card_request → 教师生成设定卡 JSON → parse_card
  1. render_episode_request    → 教师写单集 → make_episode_record（回填预估秒数/钩子兑现位置，过格式关）
  2. derive_continuations      → 由通过的单集按场景切分，规则生成『场景续写』样本（不用教师）
  3. render_edit_request       → 教师改写 → make_edit_record（带可机检的指令校验）
  4. render_judge_request      → 另一厂商的裁判打分 → judge_pass（第二道门：内容质量）

要点：教师提示词（含完整格式规范）和学生提示词（训练时 4B 模型看到的）是两回事。
学生只看到短 system + 用户输入，格式靠 SFT 学会，不靠提示词提醒。
"""
import hashlib
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

import validate_dataset as vd

SCHEMA_PATH = Path(__file__).with_name("dataset_schema.json")
TAGS = ["打脸", "反差", "揭破", "阳谋", "悬念", "反杀", "反转"]
STUDENT_SYSTEM = "你是专业短剧编剧，严格按固定的中文短剧剧本格式输出，只输出剧本正文。"

# ====================== 教师：格式规范（system） ======================
TEACHER_FORMAT_SPEC = """你是资深短剧编剧。只输出剧本正文：不要前言后语，不要解释，不要代码块围栏。必须逐字符遵守下面的格式（标点一律全角）。

【整体结构】
## 第 <<N>> 集

> 预估 0 秒 / 目标 <<T>> 秒 · 开场钩子：{一句面向观众的问题}（第 1 场第 1 拍兑现） · 结尾悬念：{本集停在哪个悬念上}
> 认领爽点：{标签}、{标签}

### 第 1 场 · {大地点·具体处}（{时间/光线/氛围}）

人物：{角色甲}、{角色乙}　道具：{道具}

{动作行}

**{角色}**（{语气}）：{台词}

……（后续场景同理，场号连续递增）

@@钩子兑现：{兑现开场钩子的那一行，原文照抄}

【格式规则】
1. 数字占位：『预估 0 秒』和『第 1 场第 1 拍兑现』里的数字照抄占位即可，脚本会按正文回填；『目标』秒数照抄。
2. 每集 1–3 场（通常 2 场）。场景名用『大地点·具体处』；括号里写时间、光线或氛围。
3. 『人物：』紧跟场景标题，列出本场所有出场角色（只有动作、没台词的也要列）。『道具』可省略；道具前是一个全角空格。
4. 每个节拍独占一行，节拍之间空一行。节拍只有两种：动作行、台词行。
5. 动作行：一句话，只写镜头拍得到的动作和画面，尽量 ≤ 30 字；不写心理，不写镜头术语。
6. 台词行：**角色**（语气）：台词。角色名加粗，括号和冒号用全角；语气 2–14 字，不加句号。台词口语化，平均约 9 字，单句尽量 ≤ 25 字。说话人必须出现在本场『人物』里；角色的内心独白用『画外音』，它不需要写进『人物』。
7. 时长：每个动作行按 2.5 秒，台词按每秒 4.5 个字符（含标点）。全集约 <<T>> 秒，大致是 18–22 个动作行加 27–50 句台词。
8. 开场钩子必须在前 3 个节拍内兑现；『开场钩子』『结尾悬念』写成面向观众的疑问或悬念，不剧透答案。最后一个节拍停在悬念上，不要收束。
9. 正文结束后空一行，单独写一行 @@钩子兑现：，后面原样抄下兑现钩子的那一行（动作行或台词行均可，抄正文，不抄格式标记也行）。
10. 原创：不得使用现实人物，不得借用已有影视、小说、游戏的人物与情节；不用 emoji，不用上面以外的 markdown。

【格式示例（只示范格式，内容勿模仿）】
### 第 1 场 · 写字楼·十九层会议室（深夜·冷白灯）

人物：许青、周总　道具：U 盘

许青把 U 盘推到会议桌中央。

**周总**（笑，不信）：这就是你说的证据？

**许青**（很平静）：播放它，你就知道了。

周总伸手去拿 U 盘，许青先一步按住。
"""

SCOPE_EPISODE = "\n【本次输出范围】输出修改后的完整一集，结构与上面完全一致（含集标题、两行概要、@@钩子兑现 行）。"
SCOPE_SCENE = "\n【本次输出范围】只输出被改写的这一场：以『### 第 K 场 · 』开头，场号保持不变；不输出集标题、概要行和 @@钩子兑现 行。"


def teacher_system(n=1, target=120, scope="episode"):
    s = TEACHER_FORMAT_SPEC.replace("<<N>>", str(n)).replace("<<T>>", str(target))
    return s + (SCOPE_SCENE if scope == "scenes" else SCOPE_EPISODE if scope == "edit_episode" else "")


# ====================== 阶段 0：设定卡 ======================
GENRES = ["古风修仙", "现代职场", "都市异能", "悬疑推理", "校园青春", "家庭伦理", "末世求生", "民国谍战", "豪门恩怨", "年代乡村", "科幻星际", "宫廷权谋"]
DEVICES = ["身份被所有人误会", "重生后抢先改写结局", "弱者其实是隐藏的最强者", "收到来自未来的消息", "被迫与仇人合作",
           "一件道具牵出三代旧账", "两人互换身份", "最信任的人其实是幕后黑手", "七十二小时倒计时", "替身被正主撞见"]
RELATIONS = ["师兄妹", "婆媳", "上下级", "继兄妹", "前任与现任", "师徒", "室友", "宿敌"]
CARD_SKELETON = ('{"title":"","genre":"","logline":"一句话设定（≤40字）","world":"世界观与规则（≤100字）",'
                 '"characters":[{"name":"","role":"","traits":"","motivation":"","speech_style":""}],'
                 '"series_arc":"全剧主线（≤120字）",'
                 '"episodes":[{"index":1,"beats":"","hook":"","ending":"","tags":["打脸"]}]}')


def render_story_card_request(seed, n_eps=6, target=120, avoid=()):
    """seed 决定类型/反差/关系的组合，保证不同作品之间题材分散；avoid 填已用过的人名、专有名词。"""
    r = random.Random(seed)
    genre, device, rel = r.choice(GENRES), r.choice(DEVICES), r.choice(RELATIONS)
    avoid_txt = f"\n6. 不得使用这些已用过的名字或设定：{'、'.join(avoid)}。" if avoid else ""
    user = (f"请为一部原创中文竖屏短剧设计『设定卡』，共 {n_eps} 集，每集约 {target} 秒。\n"
            f"本次随机种子（必须体现）：类型={genre}；核心反差={device}；主要关系={rel}。\n\n要求：\n"
            "1. 人物 3–5 个，名字 2–3 字且互不重复，避开常见网文烂名；不得使用现实人物，不得借用已有作品的人物、专有名词与情节。\n"
            "2. 每集 beats（80–150 字）写清本集发生的事，且必须含一个明确转折；hook 是面向观众的一句问题（20–45 字）；"
            "ending 写本集停在哪个悬念上（20–60 字）；tags 取 1–2 个，只能从：" + "、".join(TAGS) + "。\n"
            "3. 相邻两集：上一集的 ending 要自然引出下一集的 hook。\n"
            "4. 第 1 集的开场就要有强画面，能在 10 秒内抓住人。\n"
            "5. 只输出合法 JSON，不要解释，不要代码块围栏。" + avoid_txt + "\n\nJSON 结构：\n" + CARD_SKELETON)
    return {"work_id": f"w{seed:05d}", "target_seconds": target,
            "messages": [{"role": "system", "content": "你是短剧策划，只输出合法 JSON。"}, {"role": "user", "content": user}]}


def _strip_fence(t):
    t = t.replace("\r\n", "\n").strip()
    return re.sub(r"^```[a-zA-Z]*\n|\n```$", "", t).strip()


def parse_card(raw, work_id, target=120, n_eps=6):
    errs = []
    try:
        card = json.loads(_strip_fence(raw))
    except json.JSONDecodeError as e:
        return None, [f"设定卡不是合法 JSON：{e}"]
    for k in ("title", "genre", "logline", "world", "characters", "series_arc", "episodes"):
        if not card.get(k):
            errs.append(f"缺少字段 {k}")
    if errs:
        return None, errs
    if not 3 <= len(card["characters"]) <= 5:
        errs.append("人物数应为 3–5")
    names = [c.get("name", "") for c in card["characters"]]
    if len(set(names)) != len(names) or not all(names):
        errs.append("人物名为空或重复")
    for c in card["characters"]:
        for k in ("name", "role", "traits", "motivation", "speech_style"):
            if not c.get(k):
                errs.append(f"人物 {c.get('name')} 缺少 {k}")
    if len(card["episodes"]) != n_eps:
        errs.append(f"应有 {n_eps} 集，实际 {len(card['episodes'])}")
    for i, e in enumerate(card["episodes"], 1):
        if e.get("index") != i or not all(e.get(k) for k in ("beats", "hook", "ending", "tags")):
            errs.append(f"第 {i} 集字段不全或 index 不对")
        elif not set(e["tags"]) <= set(TAGS) or not 1 <= len(e["tags"]) <= 2:
            errs.append(f"第 {i} 集 tags 不合规：{e['tags']}")
    if errs:
        return None, errs
    card["work_id"], card["target_seconds"] = work_id, target
    return card, []


# ====================== 阶段 1：单集生成 ======================
VARIANTS = ["请按短剧剧本格式写第 {n} 集（目标 {T} 秒）。", "根据下面的设定，写出第 {n} 集短剧剧本，时长约 {T} 秒。",
            "把第 {n} 集写成可拍的短剧剧本，目标 {T} 秒，按固定格式输出。", "写第 {n} 集剧本，约 {T} 秒。"]


def auto_recap(card, n):
    return "；".join(f"第{i}集：{card['episodes'][i - 1]['beats']}" for i in range(1, n)) or "无（开篇）"


def story_block(card, n, recap=None):
    ep = card["episodes"][n - 1]
    chars = "\n".join(f"- {c['name']}：{c['role']}；性格：{c['traits']}；动机：{c['motivation']}；说话：{c['speech_style']}"
                      for c in card["characters"])
    return "\n".join([f"【作品】{card['title']}｜{card['genre']}", f"【设定】{card['logline']} {card['world']}",
                      f"【人物】\n{chars}", f"【全剧主线】{card['series_arc']}", f"【前情】{recap or auto_recap(card, n)}",
                      f"【本集任务】{ep['beats']}", f"【开场钩子】{ep['hook']}", f"【结尾悬念】{ep['ending']}",
                      f"【认领爽点】{'、'.join(ep['tags'])}"])


def render_episode_request(card, n, recap=None):
    """返回 teacher（发给教师的 messages）、student（训练样本里的 prompt）、constraints（写进 meta）。
    教师和学生的 user 消息完全相同；区别只在 system：教师带完整格式规范，学生只有一句话。"""
    T = card.get("target_seconds", 120)
    variant = random.Random(f"{card['work_id']}-{n}").choice(VARIANTS).format(n=n, T=T)
    user = f"{variant}\n\n{story_block(card, n, recap)}"
    return {"teacher": [{"role": "system", "content": teacher_system(n, T)}, {"role": "user", "content": user}],
            "student": [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": user}],
            "constraints": {"episode_index": n, "target_seconds": T, "required_tags": list(card["episodes"][n - 1]["tags"])}}


TRAILER = re.compile(r"\n+@@钩子兑现：\s*(.+?)\s*$", re.S)


def finalize_episode(raw):
    """教师原始输出 → (正文, 错误)。去围栏和前言；按『@@钩子兑现』引用的那一行回填第 K 场第 M 拍；回填预估秒数。
    LLM 数不准行数和秒数，这两处一律交给脚本。"""
    t = _strip_fence(raw)
    m = TRAILER.search(t)
    if not m:
        return None, ["缺少『@@钩子兑现：』结尾行"]
    quote, t = m[1].strip(), t[:m.start()].strip()
    i = t.find("## 第")
    if i < 0:
        return None, ["找不到『## 第 N 集』"]
    t = t[i:] + "\n"
    rep = vd.Report()
    ep = vd.parse_episode(list(enumerate(t.split("\n"), 1)), rep, "ep")
    if not ep:
        return None, rep.errors
    md = vd.RE_DLG.match(quote)
    q = md[3].strip() if md else quote
    for sc in ep["scenes"]:
        for j, (_, _, txt) in enumerate(sc["beats"], 1):
            if q and (q in txt or (len(txt) >= 4 and txt in quote)):
                t = re.sub(r"（第 \d+ 场第 \d+ 拍兑现）", f"（第 {sc['num']} 场第 {j} 拍兑现）", t, count=1)
                return vd.fix_est(t), []
    return None, [f"『@@钩子兑现』引用的行在正文里找不到：{quote[:30]}"]


# ====================== 记录组装 ======================
_SCHEMA = {}
_ARGS = SimpleNamespace(allow_round2=False, fix_est=True)


def split_of(work_id):
    """按 work_id 哈希确定划分（约 80/10/10），同一作品的所有样本必然落在同一划分，泄漏在源头杜绝。"""
    h = int(hashlib.md5(work_id.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "validation" if h == 1 else "train"


def make_record(student_msgs, completion, *, id, work_id, task_type, output_kind, source, teacher=None,
                review_status="auto_passed", parent_id=None, constraints=None, provenance="original_synthetic"):
    """返回 (记录或 None, errors, warns)。None 表示格式关没过，应丢弃或重新生成。"""
    if not _SCHEMA:
        _SCHEMA.update(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    meta = dict(id=id, work_id=work_id, task_type=task_type, output_kind=output_kind, split=split_of(work_id), source=source,
                provenance=provenance, review_status=review_status, language="zh")
    for k, v in (("teacher", teacher), ("parent_id", parent_id), ("constraints", constraints)):
        if v:
            meta[k] = v
    rec = {"prompt": student_msgs, "completion": [{"role": "assistant", "content": completion}], "meta": meta}
    rep = vd.check_record(rec, _SCHEMA, _ARGS, id)
    rec.pop("_stats", None)
    return (None if rep.errors else rec), rep.errors, rep.warns


def make_episode_record(card, n, raw, teacher):
    req = render_episode_request(card, n)
    text, errs = finalize_episode(raw)
    if errs:
        return None, errs, []
    return make_record(req["student"], text, id=f"{card['work_id']}_ep{n}", work_id=card["work_id"],
                       task_type="episode_generation", output_kind="episode", source="teacher_generated",
                       teacher=teacher, constraints=req["constraints"])


# ====================== 阶段 2：场景续写（规则生成，不用教师） ======================
def derive_continuations(rec, min_beats=3):
    """通过校验的单集 → 若干『场景续写』样本：给定第 1..k 场，目标是原文第 k+1 场。标签是现成的，零教师成本。
    目标场少于 min_beats 个节拍的（一两句过场）不出样本，没有训练价值。"""
    meta = rec["meta"]
    if meta["task_type"] != "episode_generation":
        return []
    parts = re.split(r"(?m)^(?=### 第 \d+ 场)", rec["completion"][0]["content"])[1:]
    block = rec["prompt"][-1]["content"].split("\n\n", 1)[1]
    out = []
    for k in range(1, len(parts)):
        if len(re.split(r"\n\s*\n", parts[k].strip())) - 2 < min_beats:
            continue
        done = "".join(parts[:k]).strip()
        user = f"请续写下一场（第 {k + 1} 场），保持人物与情节一致，只输出这一场。\n\n{block}\n\n【已写内容】\n{done}"
        r, errs, _ = make_record([{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": user}],
                                 parts[k].strip() + "\n", id=f"{meta['id']}_cont{k}", work_id=meta["work_id"],
                                 task_type="scene_continuation", output_kind="scenes", source="rule_generated",
                                 review_status=meta["review_status"], parent_id=meta["id"])
        if r:
            out.append(r)
    return out


# ====================== 阶段 3：改写 / 指令修改（教师） ======================
EDIT_CATALOG = {   # 指令模板；scopes: episode=整集修改(instruction_edit)，scenes=单场改写(scene_rewrite)
    "compress_time": ("把整集压缩到约 {sec} 秒：删减次要对白与动作，保留主线、钩子和结尾悬念；『目标』改为 {sec} 秒。", ["episode"]),
    "add_prop": ("加入道具「{prop}」，让它推动一次转折；写进『道具』，并至少在两个节拍里出现。", ["episode", "scenes"]),
    "remove_character": ("删去角色「{name}」，剧情走向不变；必要信息改由其他角色说出。", ["episode", "scenes"]),
    "add_voiceover": ("加入 {n} 段『画外音』（角色内心独白），其余剧情不变。", ["episode", "scenes"]),
    "subtext": ("让对白更有潜台词：少直说情绪，多用动作和反问表达；事件和人物关系不变。", ["episode", "scenes"]),
    "sharpen_conflict": ("加强人物之间的冲突：至少增加一次明确的对抗或误会，保持原有结尾。", ["episode", "scenes"]),
    "voice": ("让「{name}」的说话方式更{style}，其他角色的台词尽量不动。", ["episode", "scenes"]),
}
MACHINE_CHECKED = {"compress_time", "add_prop", "remove_character", "add_voiceover"}   # 其余交给裁判


def render_edit_request(parent, kind, params=None, scene_no=None):
    """parent: 已通过校验的单集记录。scene_no=None → 整集修改；否则改写该场。"""
    params, scope = params or {}, "episode" if scene_no is None else "scenes"
    tmpl, scopes = EDIT_CATALOG[kind]
    if scope not in scopes:
        raise ValueError(f"{kind} 不支持 {scope}")
    meta, text = parent["meta"], parent["completion"][0]["content"]
    instr = tmpl.format(**params)
    n, T = (meta.get("constraints") or {}).get("episode_index", 1), (meta.get("constraints") or {}).get("target_seconds", 120)
    if scope == "scenes":
        parts = re.split(r"(?m)^(?=### 第 \d+ 场)", text)[1:]
        original = parts[scene_no - 1].strip() + "\n"
        user = f"请改写下面这一场，保持格式、场号和人物不变：{instr}\n\n【原文】\n{original}"
    else:
        original = text
        user = f"请按指令修改下面的整集剧本，保持格式不变：{instr}\n\n【原文】\n{original}"
    if kind == "compress_time":
        T = params["sec"]
    return {"teacher": [{"role": "system", "content": teacher_system(n, T, "edit_episode" if scope == "episode" else "scenes")},
                        {"role": "user", "content": user}],
            "student": [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": user}],
            "parent": parent, "kind": kind, "params": params, "scope": scope, "original": original, "scene_no": scene_no,
            "constraints": ({"episode_index": n, "target_seconds": T} if scope == "episode" else None)}


def check_edit(req, text):
    """可机检的指令校验。返回错误列表。"""
    errs, kind, p = [], req["kind"], req["params"]
    if text.strip() == req["original"].strip():
        errs.append("教师没有做任何修改")
    if kind == "add_prop":
        scenes = vd.parse_body(list(enumerate(text.split("\n"), 1)), vd.Report(), "chk")
        if not any(p["prop"] in sc["props"] for sc in scenes):
            errs.append(f"道具「{p['prop']}」未写入『道具』字段")
        if sum(p["prop"] in b[2] for sc in scenes for b in sc["beats"]) < 2:
            errs.append(f"道具「{p['prop']}」出现在节拍里不足 2 次")
    elif kind == "remove_character" and p["name"] in text:
        errs.append(f"角色「{p['name']}」仍出现在文本中")
    elif kind == "add_voiceover" and len(re.findall(r"(?m)^\*\*画外音\*\*", text)) < p["n"]:
        errs.append(f"画外音不足 {p['n']} 段")
    return errs


def make_edit_record(req, raw, teacher):
    parent = req["parent"]
    if req["scope"] == "episode":
        text, errs = finalize_episode(raw)
    else:
        t = _strip_fence(raw)
        i = t.find("### 第")
        text, errs = (t[i:].strip() + "\n", []) if i >= 0 else (None, ["找不到『### 第 K 场』"])
    if errs:
        return None, errs, []
    errs = check_edit(req, text)
    if errs:
        return None, errs, []
    sfx = req["kind"] + ("" if req["scene_no"] is None else f"_s{req['scene_no']}")
    return make_record(req["student"], text, id=f"{parent['meta']['id']}_{sfx}", work_id=parent["meta"]["work_id"],
                       task_type="instruction_edit" if req["scope"] == "episode" else "scene_rewrite",
                       output_kind="episode" if req["scope"] == "episode" else "scenes", source="teacher_generated",
                       teacher=teacher, parent_id=parent["meta"]["id"], constraints=req["constraints"])


# ====================== 阶段 4：裁判（第二道门，内容质量） ======================
JUDGE_SYSTEM = """你是严格的短剧编审。对照【设定】【人物】【任务/指令】评审给定剧本，只输出 JSON。
评分 1–5，3 分=及格可用，4 分=好，5 分=少见的出色；不要因为文笔华丽或篇幅长而加分，只看是否符合设定与指令、是否好看。
维度：人物一致（行为与说话方式是否符合设定，有无 OOC）；情节逻辑（时间线、因果、道具与伏笔是否自洽）；
冲突与转折（有没有真正的对抗和反转，而不是空转）；对白自然（口语化、有潜台词、人物之间说话有区别）；钩子与悬念（开场抓人、结尾悬而未决）。
『致命问题』列出会让这条样本不能入库的硬伤：与设定矛盾、时间线或人物行为自相矛盾、未完成指令、抄袭既有作品的明显痕迹、内容不当；没有就给空数组。
JSON 结构：{"人物一致":0,"情节逻辑":0,"冲突与转折":0,"对白自然":0,"钩子与悬念":0,"致命问题":[],"理由":"一句话"}"""
JUDGE_KEYS = ["人物一致", "情节逻辑", "冲突与转折", "对白自然", "钩子与悬念"]


def render_judge_request(rec):
    task = rec["prompt"][-1]["content"]
    user = f"【任务/输入】\n{task}\n\n【待评剧本】\n{rec['completion'][0]['content']}"
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def judge_pass(raw, min_each=3, min_mean=3.8):
    """阈值只是起点：先人工标 30 条，对照裁判分数再校准。裁判应与教师来自不同厂商。"""
    try:
        d = json.loads(_strip_fence(raw))
        sc = [float(d[k]) for k in JUDGE_KEYS]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return False, None
    ok = not d.get("致命问题") and min(sc) >= min_each and sum(sc) / len(sc) >= min_mean
    return ok, d


def demo():
    card_req = render_story_card_request(7)
    print("=== 设定卡请求（user）===\n" + card_req["messages"][1]["content"])
    print("\n=== 教师 system（单集，节选）===\n" + teacher_system(1, 120)[:700] + "\n……")


if __name__ == "__main__":
    demo()
