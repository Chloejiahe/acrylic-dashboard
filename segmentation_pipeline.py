import re
from pathlib import Path

import numpy as np
import pandas as pd

from feature_dic import FEATURE_DIC

try:
    from textblob import TextBlob
except Exception:  # pragma: no cover
    TextBlob = None


# =========================================================
# 配置
# =========================================================
ENABLE_POLARITY = False   # True = 计算 TextBlob 句子极性；False = 不算，速度更快
SAVE_EXCEL = False        # True = 额外写出 xlsx；False = 只写 parquet/csv，速度更快


# =========================================================
# 路径
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "data" / "raw" / "评论总表.xlsx"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 文本工具
# =========================================================
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[/_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_kw(kw):
    kw = str(kw).lower().strip()
    kw = re.sub(r"[/_\-]+", " ", kw)
    kw = re.sub(r"[^a-z0-9\s]", " ", kw)
    kw = re.sub(r"\s+", " ", kw)
    return kw.strip()


def keyword_found_norm(text_norm, kw_norm):
    if not kw_norm:
        return False

    if " " in kw_norm:
        return kw_norm in text_norm

    padded_text = f" {text_norm} "
    padded_kw = f" {kw_norm} "
    return padded_kw in padded_text


def unique_keyword_hits_norm(text_norm, keywords_norm):
    hits = []
    seen = set()

    for kw_norm in keywords_norm:
        if not kw_norm:
            continue
        if kw_norm in seen:
            continue
        if keyword_found_norm(text_norm, kw_norm):
            hits.append(kw_norm)
            seen.add(kw_norm)

    return hits


def unique_keyword_hits(text, keywords):
    text_norm = normalize_text(text)
    keywords_norm = [normalize_kw(kw) for kw in keywords]
    return unique_keyword_hits_norm(text_norm, keywords_norm)


def make_keyword_hint(keywords, max_n=4):
    vals = []
    seen = set()
    for kw in keywords:
        kw_norm = normalize_kw(kw)
        if kw_norm and kw_norm not in seen:
            vals.append(kw_norm)
            seen.add(kw_norm)
    return ", ".join(vals[:max_n])


def split_sentences(text):
    """轻量句子切分，避免依赖 nltk 资源下载。"""
    if pd.isna(text):
        return []

    text = str(text).strip()
    if not text:
        return []

    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    parts = re.split(r"(?<=[\.!?;。！？；])\s+", text)
    parts = [p.strip() for p in parts if str(p).strip()]

    if not parts:
        return [text]
    return parts


def get_sentence_polarity(sentence):
    if not ENABLE_POLARITY:
        return 0.0
    if not sentence:
        return 0.0
    if TextBlob is None:
        return 0.0
    try:
        return float(TextBlob(str(sentence)).sentiment.polarity)
    except Exception:
        return 0.0


def choose_analysis_text(row):
    """优先用英文翻译列做规则分析，保留原评论做展示。"""
    for col in ["Content"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return str(row.get("Content", "")).strip()


def save_outputs(df_obj, base_name):
    parquet_path = PROCESSED_DIR / f"{base_name}.parquet"
    csv_path = PROCESSED_DIR / f"{base_name}.csv"
    df_obj.to_parquet(parquet_path, index=False)
    df_obj.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if SAVE_EXCEL:
        excel_path = PROCESSED_DIR / f"{base_name}.xlsx"
        df_obj.to_excel(excel_path, index=False)



# =========================================================
# 主分群：仍然以“怎么用”为主，但扩大词覆盖
# =========================================================
SEGMENT_RULES = [
    {
        "segment": "儿童/课堂使用群",
        "core": [
            "for kids", "for my kids", "for children", "for my son", "for my daughter",
            "for my toddler", "with my children", "kids craft", "crafts for kids",
            "family craft time", "art project for kids", "preschool activities",
            "learning colors", "develop fine motor skills", "kid friendly",
            "safe for children", "safe for kids", "art class", "craft class",
            "for my students", "for the class", "classroom supplies", "student work",
            "school project", "class assignment", "teaching a class", "art education",
            "homeschooling", "homeschool", "teacher"
        ],
        "aux": [
            "kid", "kids", "child", "children", "toddler", "preschooler",
            "student", "students", "school", "teacher", "class", "classroom",
            "homeschool", "educational toy", "family fun"
        ]
    },
    {
        "segment": "多表面diy/定制群",
        "core": [
            "diy project", "craft project", "crafting", "for crafts", "arts and crafts",
            "decorating ornaments", "customizing shoes", "phone case decoration",
            "painting pumpkins", "easter egg decorating", "on glass", "on t shirt",
            "on fabric", "on plastic", "on metal", "model painting",
            "miniature painting", "painting miniatures", "warhammer painting",
            "model building", "customizing", "rock painting", "mug decoration",
            "wood signs", "wood crafts", "resin art", "resin crafts",
            "polymer clay crafts", "jewelry making", "candle making", "wreath making",
            "on wood", "on rocks", "on rock", "on stone", "on ceramic", "on mugs"
        ],
        "aux": [
            "diy", "craft", "crafts", "rock", "rocks", "wood", "glass", "ceramic",
            "mug", "fabric", "plastic", "metal", "stone", "ornament", "resin",
            "clay", "miniature", "model", "customizing"
        ]
    },
    {
        "segment": "纸面手账/装饰书写群",
        "core": [
            "calligraphy", "lettering", "hand lettering", "brush lettering",
            "journal headers", "planner headers", "writing letters", "place cards",
            "wedding invitations", "note taking", "taking notes", "study notes",
            "meeting notes", "class notes", "annotating books", "marking up documents",
            "color coding", "color code my notes", "organizing my calendar",
            "calendar planning", "making labels", "to do list", "making lists",
            "bullet journal", "journal", "journaling", "planner", "scrapbook",
            "scrapbooking", "card making", "greeting card", "gift tag"
        ],
        "aux": [
            "journal", "journaling", "planner", "lettering", "calligraphy",
            "scrapbook", "diary", "notes", "note", "labeling", "calendar",
            "headers", "card", "cards", "handwriting"
        ]
    },
    {
        "segment": "精细绘画/插画创作群",
        "core": [
            "making art", "creating art", "for my art", "art project", "fine art",
            "for drawing", "illustration", "manga", "comic art", "landscape sketch",
            "urban sketching", "artwork", "portrait drawing", "character design",
            "sketching", "botanical illustration", "still life", "figure drawing",
            "inking lines", "animal drawing", "concept art", "line art",
            "fine details", "detailed work", "intricate work"
        ],
        "aux": [
            "illustration", "illustrating", "drawing", "sketch", "sketching",
            "artwork", "portrait", "comic", "manga", "anime", "detail", "detailed",
            "precision", "character", "line art"
        ]
    },
    {
        "segment": "填色/大面积上色群",
        "core": [
            "coloring book", "coloring books", "adult coloring", "colouring book",
            "color page", "coloring pages", "adult coloring book", "color therapy",
            "mindfulness coloring", "relaxing coloring", "intricate coloring",
            "detailed coloring", "secret garden", "johanna basford", "kerby rosanes",
            "hanna karlzon", "mandalas", "mandala coloring", "color by number",
            "mystery coloring", "fill large areas", "large coverage",
            "background coloring", "filling in larger areas", "great for coloring books"
        ],
        "aux": [
            "coloring", "colouring", "mandala", "fill", "filling", "background",
            "color pages", "adult coloring", "large areas"
        ]
    },
    {
        "segment": "设计工作/专业工作群",
        "core": [
            "design work", "for my design work", "professional design", "client design",
            "design project", "fashion design", "fashion illustration", "garment design",
            "textile design", "product design", "industrial design", "product sketch",
            "rendering", "graphic design", "logo design", "layout design", "branding",
            "ui design", "ux design", "wireframing", "mockup", "architecture",
            "architectural drawing", "interior design", "floor plan", "blueprint",
            "schematics", "storyboard", "for professional work", "client work"
        ],
        "aux": [
            "design", "professional", "client", "branding", "graphic", "layout",
            "ui", "ux", "mockup", "architecture", "interior", "blueprint",
            "rendering", "storyboard"
        ]
    }
]


# =========================================================
# 画像维度
# =========================================================
ATTRIBUTE_RULES = {
    "绘画主题": {
    "人物": [
        "portrait", "portraits", "character", "characters", "figure drawing",
        "anime", "manga", "anime characters", "anime pictures",
        "face", "people", "draw people"
    ],
    "植物/自然": [
        "landscape", "nature", "garden", "tree", "trees",
        "flower", "flowers", "botanical", "floral"
    ],
    "装饰物/器物": [
        "still life", "object drawing", "ornaments", "ornament",
        "christmas ornaments", "cards", "greeting cards",
        "signs", "posters", "pumpkins", "eggs",
        "jars", "mason jars", "cups"
    ],
    "动物": [
        "animal drawing", "animal", "animals", "pet sketch",
        "pet portrait", "cat", "cats", "dog", "dogs",
        "bird", "birds"
    ]
},
"绘画场景": {
    "课堂/学习": [
        "art class", "school project", "for students", "students",
        "teacher", "classroom", "for class", "teaching",
        "homeschool", "tutorial"
    ],
    "家庭/个人": [
        "for my kids", "my kids", "for fun", "home project",
        "my daughter", "my son", "granddaughter", "grandson",
        "grandkids", "grandchildren", "niece", "nephew",
        "at home", "myself", "birthday party", "craft box"
    ],
    "户外/社区装饰": [
        "outside", "outdoor", "outdoors", "garden",
        "kindness rocks", "garden rocks", "yard"
    ],
    "工作室/商业制作": [
        "portfolio project", "client work", "professional work",
        "studio", "mural", "commission", "business",
        "for my business", "etsy", "craft fair"
    ]
},
"购买动机": {
    "入门学习": [
        "starter kit", "starter set", "for beginners", "beginner friendly",
        "easy to learn", "just starting out", "new to art", "first time trying",
        "for practice", "learning a new skill", "beginners", "first time",
        "first time using", "my first set", "new to paint pens",
        "great for beginners", "easy to use", "easy to prime"
    ],
    "提升技能": [
        "challenge myself", "step up my game", "refine my skills",
        "take my art to the next level", "unlock new techniques",
        "advanced techniques", "improve my drawing",
        "better control", "fine details", "detailed work", "precision work"
    ],
    "专业产出": [
        "professional grade", "for professional work", "client work",
        "serious tool", "artist grade", "pro grade", "reliable for work",
        "studio", "portrait artist", "mural", "commission",
        "for my business", "sign making"
    ],
    "兴趣爱好": [
        "for my hobby", "for fun", "spark creativity", "express myself",
        "wanted to try", "get back into art", "artistic exploration",
        "hobby", "relaxing hobby", "creative", "crafting",
        "love to draw", "draw for fun"
    ],
    "礼品需求": [
        "as a gift", "for a present", "gift for someone", "birthday gift",
        "christmas gift", "holiday gift", "giftable",
        "gift for", "as gifts", "gift idea", "for christmas",
        "christmas present", "bought as a gift", "perfect gift"
    ],
    "品牌信任": [
        "trusted brand", "good reputation", "well known brand",
        "always reliable", "go to brand", "brand loyalty",
        "love this brand", "brand name", "posca", "sharpie"
    ],
    "社交/媒体种草": [
        "recommended by", "my friend recommended", "my teacher recommended",
        "saw good reviews", "saw it on social media", "tiktok made me buy it",
        "saw it on instagram", "youtube review", "influencer recommended",
        "pinterest"
    ],
    "便携性需求": [
        "portable", "on the go", "easy to carry", "travel set", "compact",
        "lightweight", "fits in my bag", "take it anywhere",
        "small enough", "carry around", "travel case"
    ],
    "多功能需求": [
        "multi purpose", "all in one", "works for everything",
        "works on multiple surfaces", "good for many things", "one set for all my needs",
        "versatile", "different surfaces", "various surfaces",
        "any surface", "all surfaces", "many surfaces"
    ],
    "压力缓解/情绪调节": [
        "stress relief", "for relaxation", "to relax", "calming", "art therapy",
        "therapeutic", "to unwind", "meditative", "calms me down",
        "relax", "relaxing", "therapy"
    ]
},
"使用对象": {
    "儿童": [
        "kid", "kids", "child", "children", "toddler", "baby",
        "preschooler", "little one", "for my son", "for my daughter",
        "grandson", "granddaughter", "for my grandson", "for my granddaughter",
        "grandkids", "grandchildren", "niece", "nephew", "my kids"
    ],
    "青少年/学生": [
        "teen", "teenager", "adolescent", "high school", "college student",
        "university student", "student", "students", "art student",
        "teenage son", "teenage daughter"
    ],
    "成人": [
        "adult coloring", "for my hobby", "for fun", "journaling",
        "crafting", "relaxing hobby", "artist", "crafters", "hobbyist"
    ],
    "老年人": [
        "senior", "elderly", "retired", "grandparent", "grandfather",
        "grandmother", "golden years", "grandma", "grandpa"
    ]
},
"介质偏好": {
    "纸张/卡纸": [
        "on paper", "paper", "marker paper", "bristol", "cardstock",
        "watercolor paper", "mixed media paper", "black cardstock"
    ],
    "深色纸": [
        "black paper", "dark paper", "kraft paper", "colored paper"
    ],
    "木材": [
        "on wood", "wood", "on wooden", "wood crafts", "wood projects",
        "wood slice", "wood slices", "wood signs", "wooden signs",
        "wooden ornaments", "pallet", "pallets"
    ],
    "石头/岩石": [
        "on rock", "on rocks", "rock painting", "painting rocks",
        "paint rocks", "paint on rocks", "rock", "rocks",
        "on stone", "stone", "stones", "stone painting",
        "painted stones", "pebbles", "kindness rocks"
    ],
    "玻璃": [
        "on glass", "glass", "glass art", "wine glass",
        "mirror", "jars", "mason jars"
    ],
    "陶瓷": [
        "on ceramic", "ceramic", "porcelain", "mug", "mugs",
        "mug decoration", "ceramic mug"
    ],
    "画布": [
        "on canvas", "canvas", "canvas board", "stretched canvas"
    ],
    "布料": [
        "on fabric", "fabric", "on t shirt", "t shirt",
        "on textile", "textile", "on denim", "denim"
    ],
    "鞋类/皮革": [
        "shoes", "canvas shoes", "sneakers", "leather"
    ],
    "塑料": [
        "on plastic", "plastic", "plastic models", "plastic ornaments",
        "plastic cups", "3d printed"
    ],
    "金属": [
        "on metal", "metal", "on aluminum", "aluminum"
    ],
    "树脂/黏土": [
        "on resin", "resin", "epoxy", "polymer clay"
    ],
    "黑板/光滑特殊表面": [
        "chalkboard", "blackboard"
    ]
},
"产品期望": {
    "颜色表现": [
        "vibrant colors", "rich colors", "deep saturation", "true to color",
        "color accuracy", "pigmented", "high quality pigment", "color range",
        "bright colors", "vivid colors", "variety of colors",
        "color selection", "wide range of colors",
        "beautiful colors", "nice colors", "lots of colors",
        "good variety of colours", "great color selection", "bright and vivid"
    ],
    "覆盖显色": [
        "opaque", "opaque coverage", "covers well", "one coat",
        "shows up on black", "good opacity", "good coverage",
        "cover well", "show up well", "coverage", "solid coverage"
    ],
    "流畅出墨": [
        "smooth flow", "flows well", "smoothly", "consistent flow",
        "no skipping", "good ink flow", "easy to control",
        "ink flow", "flows smoothly", "smooth application",
        "writes smoothly", "smooth writing", "easy flow", "went on smooth", "glides"
    ],
    "耐用寿命": [
        "long lasting", "durable tip", "built to last", "workhorse",
        "holds up to heavy use", "great longevity",
        "lasts a long time", "last long", "holds up well",
        "still work", "lasted", "didn t dry out"
    ],
    "安全/气味": [
        "non toxic", "safe for children", "odorless", "no smell",
        "low odor", "xylene free", "no odor", "safe for kids",
        "doesn t smell", "strong smell", "chemical smell"
    ],
    "包装/收纳": [
        "beautiful packaging", "nice box", "storage case", "travel case",
        "organized", "easy to store", "good presentation",
        "carrying case", "storage box", "packaging", "box", "case",
        "individually wrapped"
    ],
    "便携性": [
        "portable", "compact", "lightweight", "travel friendly",
        "fits in my bag", "easy to carry", "small enough", "carry around"
    ],
    "性价比": [
        "good value", "great price", "affordable", "good deal",
        "cheap but good", "cost effective", "worth the money",
        "value for the money", "price point", "worth every penny"
    ],
    "多功能性": [
        "works on multiple surfaces", "multi purpose", "all in one",
        "use it for everything", "good for many things",
        "different surfaces", "various surfaces", "any surface",
        "almost any surface", "many surfaces", "all surfaces", "versatile"
    ],
    "效率/快干": [
        "quick drying", "fast drying", "dries instantly", "saves me time",
        "work faster", "improves my workflow", "hassle free",
        "dries fast", "quick dry", "dry quickly"
    ]
  }
}

# =========================================================
# bundle insight：句子级证据
# =========================================================
BUNDLE_PRODUCT_DIC = {
    "纸质媒介 (Paper & Pads)": {
        "黑卡纸/本": ["black paper", "black cardstock", "dark paper", "black notebook", "black pad"],
        "绘本/写生本": ["sketchbook", "sketch pad", "drawing book", "art journal", "mixed media pad"],
        "重磅马克笔纸": ["marker paper", "heavyweight paper", "smooth cardstock", "160gsm", "200gsm", "thick paper"],
        "涂鸦板/卡片": ["flashcards", "index cards", "diy cards", "tags"],
        "水彩纸/多媒体纸": ["watercolor paper", "textured paper", "cold press", "mixed media paper"],
        "黑色便利贴": ["black sticky notes", "black post its", "dark sticky notes"]
    },
    "涂色与创作 (Coloring & Greeting)": {
        "成人涂色书": ["coloring book", "adult coloring", "mandala book", "therapy coloring"],
        "贺卡/信封": ["greeting cards", "envelopes", "invitations", "blank cards", "thank you cards"],
        "明信片": ["postcard", "postcards", "mailing cards", "blank postcards", "postal cards"],
        "空白标签": ["gift tags", "label tags", "price tags", "hanging tags"]
    },
    "勾线与细节 (Detailing & Outlining)": {
        "极细勾线笔": ["fineliner", "micro tip", "0 5mm pen", "ultra fine pen", "detail pen", "outline pen"],
        "铅笔/橡皮": ["graphite pencil", "pencil", "sketching pencil", "kneaded eraser", "rubber", "electric eraser"]
    },
    "表面保护 (Finishing & Protection)": {
        "亮油/保护喷雾": ["varnish", "sealer", "glossy spray", "fixative", "top coat", "clear coat"],
        "密封胶": ["sealant", "mod podge", "acrylic sealer", "glue sealer"],
        "遮蔽胶带": ["masking tape", "washi tape", "painter s tape", "decorative tape"]
    },
    "辅助与创意 (Tools & Accessories)": {
        "镂空模板": ["stencils", "drawing template", "alphabet stencil", "pattern stencil"],
        "便携笔袋/盒": ["carrying case", "storage bag", "organizer pouch", "holder", "pen stand", "acrylic holder"],
        "火漆/装饰": ["wax seal", "sealing wax", "stamps", "gold leaf"],
        "调色/混色": ["mixing palette", "paint tray", "dotting tools", "blending sponge"],
        "贴纸/胶水": ["stickers", "glue pen", "adhesive", "decals"]
    }
}

# =========================================================
# 预处理索引（加速）
# =========================================================
def dedupe_norm_keywords(keywords):
    out = []
    seen = set()
    for kw in keywords:
        kw_norm = normalize_kw(kw)
        if kw_norm and kw_norm not in seen:
            out.append(kw_norm)
            seen.add(kw_norm)
    return out


def prepare_segment_rules(segment_rules):
    prepared = []
    for i, rule in enumerate(segment_rules):
        prepared.append({
            "segment": rule["segment"],
            "priority": i,
            "core_norm": dedupe_norm_keywords(rule["core"]),
            "aux_norm": dedupe_norm_keywords(rule["aux"]),
        })
    return prepared


def prepare_attribute_rules(attribute_rules):
    prepared = {}
    for dim_name, label_map in attribute_rules.items():
        prepared[dim_name] = {}
        for label, keywords in label_map.items():
            prepared[dim_name][label] = dedupe_norm_keywords(keywords)
    return prepared


def prepare_feature_index(feature_dic):
    feature_index = {}
    for dim_name, tag_map in feature_dic.items():
        neg_list = []
        pos_list = []

        for tag, keywords in tag_map.items():
            norm_keywords = dedupe_norm_keywords(keywords)

            if "负面" in tag or "不满" in tag:
                neg_list.append((tag, norm_keywords))
            elif "正面" in tag:
                pos_list.append((tag, norm_keywords))

        feature_index[dim_name] = {
            "neg": neg_list,
            "pos": pos_list
        }
    return feature_index


def prepare_bundle_index(bundle_dic):
    bundle_index = []
    for big_cat, sub_dict in bundle_dic.items():
        for sub_item, keywords in sub_dict.items():
            norm_keywords = dedupe_norm_keywords(keywords)
            bundle_index.append((big_cat, sub_item, norm_keywords))
    return bundle_index


PREPARED_SEGMENT_RULES = prepare_segment_rules(SEGMENT_RULES)
PREPARED_ATTRIBUTE_RULES = prepare_attribute_rules(ATTRIBUTE_RULES)
FEATURE_INDEX = prepare_feature_index(FEATURE_DIC)
BUNDLE_INDEX = prepare_bundle_index(BUNDLE_PRODUCT_DIC)


# =========================================================
# 主分群判定
# =========================================================
def assign_primary_segment(text_norm):
    candidates = []

    for rule in PREPARED_SEGMENT_RULES:
        core_hits = unique_keyword_hits_norm(text_norm, rule["core_norm"])
        aux_hits = unique_keyword_hits_norm(text_norm, rule["aux_norm"])
        total_hits = len(set(core_hits + aux_hits))
        score = len(core_hits) * 3 + len(aux_hits)

        candidates.append({
            "segment": rule["segment"],
            "priority": rule["priority"],
            "core_hits": len(core_hits),
            "aux_hits": len(aux_hits),
            "total_hits": total_hits,
            "score": score,
            "core_evidence": " | ".join(core_hits[:12]),
            "aux_evidence": " | ".join(aux_hits[:12])
        })

    ranked = sorted(
        candidates,
        key=lambda x: (x["core_hits"], x["total_hits"], x["score"], -x["priority"]),
        reverse=True
    )

    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    second_score = second["score"] if second else 0
    score_gap = best["score"] - second_score

    is_noise = int(best["total_hits"] == 0)

    if best["core_hits"] >= 1 or best["total_hits"] >= 3:
        confidence = "high"
    elif best["total_hits"] >= 2:
        confidence = "medium"
    elif best["total_hits"] == 1:
        confidence = "low"
    else:
        confidence = "none"

    return {
        "primary_segment": None if is_noise else best["segment"],
        "segment_core_hits": best["core_hits"],
        "segment_aux_hits": best["aux_hits"],
        "segment_total_hits": best["total_hits"],
        "segment_score": best["score"],
        "segment_score_gap": score_gap,
        "segment_confidence": confidence,
        "segment_core_evidence": best["core_evidence"],
        "segment_aux_evidence": best["aux_evidence"],
        "is_noise": is_noise
    }

# =========================================================
# attribute 标签 + 每个维度的主标签 + 命中关键词提示
# =========================================================
def build_attribute_flags_and_top(text_norm):
    result = {}

    for dim_name, label_map in PREPARED_ATTRIBUTE_RULES.items():
        label_scores = {}
        label_hits_map = {}

        for label, keywords_norm in label_map.items():
            hits = unique_keyword_hits_norm(text_norm, keywords_norm)
            label_scores[label] = len(hits)
            label_hits_map[label] = hits
            result[f"ATTR__{dim_name}__{label}"] = 1 if len(hits) > 0 else 0

        valid_scores = {k: v for k, v in label_scores.items() if v > 0}

        if valid_scores:
            top_label = sorted(valid_scores.items(), key=lambda x: (-x[1], x[0]))[0][0]
            top_score = valid_scores[top_label]
            top_hits = label_hits_map[top_label]

            raw_keywords = ATTRIBUTE_RULES[dim_name][top_label]
            static_hint = make_keyword_hint(raw_keywords, max_n=4)
            matched_hint = ", ".join(top_hits[:4]) if top_hits else static_hint
            top_display = f"{top_label}（{static_hint}）"
        else:
            top_label = "未提及"
            top_score = 0
            top_hits = []
            static_hint = ""
            matched_hint = ""
            top_display = "未提及"

        result[f"ATTR_TOP__{dim_name}"] = top_label
        result[f"ATTR_TOP_SCORE__{dim_name}"] = top_score
        result[f"ATTR_TOP_HINT__{dim_name}"] = static_hint
        result[f"ATTR_TOP_MATCHED__{dim_name}"] = " | ".join(top_hits[:8])
        result[f"ATTR_TOP_MATCHED_HINT__{dim_name}"] = matched_hint
        result[f"ATTR_TOP_DISPLAY__{dim_name}"] = top_display
        result[f"ATTR_MENTIONED__{dim_name}"] = 1 if top_score > 0 else 0

    return result


def build_attribute_label_meta():
    rows = []
    for dim_name, label_map in ATTRIBUTE_RULES.items():
        for label, keywords in label_map.items():
            hint = make_keyword_hint(keywords, max_n=4)
            rows.append({
                "attribute_dimension": dim_name,
                "attribute_label": label,
                "attribute_label_hint": hint,
                "attribute_label_display": f"{label}（{hint}）" if hint else label
            })
    return pd.DataFrame(rows)


# =========================================================
# 亮点 / 痛点：句子级分析
# =========================================================
def build_feature_flags_and_quotes(row):
    flags = {}
    quote_rows = []

    raw_text = row.analysis_text_raw
    sentences = split_sentences(raw_text)

    dim_pos_hit = {dim: 0 for dim in FEATURE_INDEX.keys()}
    dim_neg_hit = {dim: 0 for dim in FEATURE_INDEX.keys()}

    for sent_idx, sentence in enumerate(sentences, start=1):
        sent_norm = normalize_text(sentence)
        if not sent_norm:
            continue

        sent_pol = None

        for dim_name, dim_rules in FEATURE_INDEX.items():
            neg_found_this_dim = False

            for tag, keywords_norm in dim_rules["neg"]:
                hits = unique_keyword_hits_norm(sent_norm, keywords_norm)
                if not hits:
                    continue

                if sent_pol is None:
                    sent_pol = get_sentence_polarity(sentence)

                neg_found_this_dim = True
                dim_neg_hit[dim_name] = 1

                quote_rows.append({
                    "review_id": row.review_id,
                    "sentence_id": f"{row.review_id}_{sent_idx}",
                    "sentence_index": sent_idx,
                    "primary_segment": row.primary_segment,
                    "Asin": getattr(row, "Asin", ""),
                    "Brand": getattr(row, "Brand", ""),
                    "Nation": getattr(row, "Nation", ""),
                    "Rating": getattr(row, "Rating", np.nan),
                    "sentence_polarity": round(sent_pol, 4),
                    "sentiment_type": "negative",
                    "feature_dimension": dim_name,
                    "feature_tag": tag,
                    "matched_keywords": " | ".join(hits[:8]),
                    "Sentence": sentence,
                    "Content": getattr(row, "Content", "")
                })

            if neg_found_this_dim:
                continue

            for tag, keywords_norm in dim_rules["pos"]:
                hits = unique_keyword_hits_norm(sent_norm, keywords_norm)
                if not hits:
                    continue

                if sent_pol is None:
                    sent_pol = get_sentence_polarity(sentence)

                dim_pos_hit[dim_name] = 1

                quote_rows.append({
                    "review_id": row.review_id,
                    "sentence_id": f"{row.review_id}_{sent_idx}",
                    "sentence_index": sent_idx,
                    "primary_segment": row.primary_segment,
                    "Asin": getattr(row, "Asin", ""),
                    "Brand": getattr(row, "Brand", ""),
                    "Nation": getattr(row, "Nation", ""),
                    "Rating": getattr(row, "Rating", np.nan),
                    "sentence_polarity": round(sent_pol, 4),
                    "sentiment_type": "positive",
                    "feature_dimension": dim_name,
                    "feature_tag": tag,
                    "matched_keywords": " | ".join(hits[:8]),
                    "Sentence": sentence,
                    "Content": getattr(row, "Content", "")
                })

    for dim_name in FEATURE_INDEX.keys():
        flags[f"NEG__{dim_name}"] = dim_neg_hit[dim_name]
        flags[f"POS__{dim_name}"] = dim_pos_hit[dim_name]

    return flags, quote_rows


# =========================================================
# bundle insight：句子级证据
# =========================================================
def build_bundle_quotes(row):
    quote_rows = []
    raw_text = row.analysis_text_raw
    sentences = split_sentences(raw_text)

    for sent_idx, sentence in enumerate(sentences, start=1):
        sent_norm = normalize_text(sentence)
        if not sent_norm:
            continue

        for big_cat, sub_item, keywords_norm in BUNDLE_INDEX:
            hits = unique_keyword_hits_norm(sent_norm, keywords_norm)
            if not hits:
                continue

            quote_rows.append({
                "review_id": row.review_id,
                "sentence_id": f"{row.review_id}_{sent_idx}",
                "sentence_index": sent_idx,
                "bundle_category": big_cat,
                "bundle_sub_item": sub_item,
                "matched_keywords": " | ".join(hits[:8]),
                "Asin": getattr(row, "Asin", ""),
                "Brand": getattr(row, "Brand", ""),
                "Rating": getattr(row, "Rating", np.nan),
                "Sentence": sentence,
                "Content": getattr(row, "Content", "")
            })

    return quote_rows

# =========================================================
# 读取数据
# =========================================================
if not RAW_FILE.exists():
    raise FileNotFoundError(f"找不到原始文件: {RAW_FILE}")

print("开始运行 segmentation pipeline（加速版）...")
print("正在读取原始 Excel ...")

df = pd.read_excel(RAW_FILE)
df.columns = [str(c).strip() for c in df.columns]

required_cols = ["Content", "Asin", "Brand", "Nation", "Rating"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Excel 缺少这些必要列: {missing}")

df = df.copy()
df["Content"] = df["Content"].fillna("").astype(str).str.strip()
df = df[df["Content"] != ""].copy()

df["review_id"] = range(1, len(df) + 1)
df["analysis_text_raw"] = df.apply(choose_analysis_text, axis=1)
df["content_lower"] = df["analysis_text_raw"].apply(normalize_text)


# =========================================================
# 主分群
# =========================================================
print("主分群中 ...")
segment_records = df["content_lower"].apply(assign_primary_segment).apply(pd.Series)
df = pd.concat([df, segment_records], axis=1)


# =========================================================
# 全量评论：句子级亮点 / 痛点 flags + 句子级原声
# =========================================================
print("进行句子级亮点 / 痛点分析中（全量评论）...")
feature_flag_rows = []
quote_rows_all = []

for row in df.itertuples(index=False):
    flags, q_rows = build_feature_flags_and_quotes(row)
    feature_flag_rows.append(flags)
    quote_rows_all.extend(q_rows)

feature_flags_df = pd.DataFrame(feature_flag_rows, index=df.index)
df = pd.concat([df, feature_flags_df], axis=1)

feature_quotes_df = pd.DataFrame(quote_rows_all)
if feature_quotes_df.empty:
    feature_quotes_df = pd.DataFrame(columns=[
        "review_id", "sentence_id", "sentence_index", "primary_segment", "Asin", "Brand",
        "Nation", "Rating", "sentence_polarity", "sentiment_type", "feature_dimension",
        "feature_tag", "matched_keywords", "Sentence", "Content"
    ])


# =========================================================
# 全量评论：Bundle Insight 句子级证据
# =========================================================
print("提取 Bundle Insight 证据中（全量评论）...")
bundle_rows_all = []
for row in df.itertuples(index=False):
    bundle_rows_all.extend(build_bundle_quotes(row))

bundle_quotes_df = pd.DataFrame(bundle_rows_all)
if bundle_quotes_df.empty:
    bundle_quotes_df = pd.DataFrame(columns=[
        "review_id", "sentence_id", "sentence_index", "bundle_category", "bundle_sub_item",
        "matched_keywords", "Asin", "Brand", "Rating", "Sentence", "Content"
    ])


# =========================================================
# 再拆 classified / noise
# =========================================================
noise_df = df[df["is_noise"] == 1].copy()
classified_df = df[df["is_noise"] == 0].copy()

if classified_df.empty:
    raise ValueError("当前规则下没有可分群评论，请检查规则或原始数据。")


# =========================================================
# 人群画像 attribute（仍然只对 classified 做）
# =========================================================
print("生成 attribute 标签中 ...")
attr_df = classified_df["content_lower"].apply(build_attribute_flags_and_top).apply(pd.Series)
classified_df = pd.concat([classified_df, attr_df], axis=1)

attribute_label_meta = build_attribute_label_meta()
print("写出 attribute_label_meta ...")
save_outputs(attribute_label_meta, "attribute_label_meta")


# =========================================================
# reviews_all（全量评论基础表，给筛选器和下半部分句子分析用）
# =========================================================
all_base_keep = [
    "review_id", "Asin", "Brand", "Nation", "Rating", "Content",
    "primary_segment", "is_noise"
]
all_optional_keep = [c for c in ["出墨方式", "Price Level"] if c in df.columns]

reviews_all = df[all_base_keep + all_optional_keep].copy()

print("写出 reviews_all ...")
save_outputs(reviews_all, "reviews_all")


# =========================================================
# reviews_segmented
# =========================================================
base_keep = [
    "review_id", "Asin", "Brand", "Nation", "Rating", "Content",
    "primary_segment", "segment_core_hits", "segment_aux_hits",
    "segment_total_hits", "segment_score", "segment_score_gap",
    "segment_confidence", "segment_core_evidence", "segment_aux_evidence"
]

optional_keep = [c for c in ["出墨方式", "Price Level"] if c in classified_df.columns]
attr_keep = [
    c for c in classified_df.columns
    if c.startswith("ATTR__")
    or c.startswith("ATTR_TOP__")
    or c.startswith("ATTR_TOP_SCORE__")
    or c.startswith("ATTR_TOP_HINT__")
    or c.startswith("ATTR_TOP_MATCHED__")
    or c.startswith("ATTR_TOP_MATCHED_HINT__")
    or c.startswith("ATTR_TOP_DISPLAY__")
    or c.startswith("ATTR_MENTIONED__")
]
feature_keep = [c for c in classified_df.columns if c.startswith("NEG__") or c.startswith("POS__")]

reviews_segmented = classified_df[base_keep + optional_keep + attr_keep + feature_keep].copy()

print("写出 reviews_segmented ...")
save_outputs(reviews_segmented, "reviews_segmented")


# =========================================================
# segment_summary
# =========================================================
segment_summary = (
    reviews_segmented.groupby("primary_segment")
    .agg(
        review_count=("review_id", "count"),
        avg_rating=("Rating", "mean")
    )
    .reset_index()
    .sort_values("review_count", ascending=False)
)
segment_summary["avg_rating"] = segment_summary["avg_rating"].round(2)
save_outputs(segment_summary, "segment_summary")


# =========================================================
# segment_profile_long
# =========================================================
attr_top_cols = [
    c for c in reviews_segmented.columns
    if c.startswith("ATTR_TOP__")
    and not c.startswith("ATTR_TOP_SCORE__")
    and not c.startswith("ATTR_TOP_HINT__")
    and not c.startswith("ATTR_TOP_MATCHED__")
    and not c.startswith("ATTR_TOP_MATCHED_HINT__")
    and not c.startswith("ATTR_TOP_DISPLAY__")
]

profile_rows = []

for col in attr_top_cols:
    dim_name = col.replace("ATTR_TOP__", "")
    temp = reviews_segmented[["review_id", "primary_segment", col]].copy()
    temp = temp.rename(columns={col: "attribute_label"})

    temp_mentioned = temp[temp["attribute_label"] != "未提及"].copy()
    if temp_mentioned.empty:
        continue

    cnt = (
        temp_mentioned.groupby(["primary_segment", "attribute_label"])["review_id"]
        .count()
        .reset_index(name="label_count")
    )
    den = (
        temp_mentioned.groupby("primary_segment")["review_id"]
        .count()
        .reset_index(name="dimension_mentioned_count")
    )
    out = cnt.merge(den, on="primary_segment", how="left")
    out["attribute_dimension"] = dim_name
    out["pct_within_dimension"] = out["label_count"] / out["dimension_mentioned_count"]
    profile_rows.append(out)

segment_profile_long = (
    pd.concat(profile_rows, ignore_index=True)
    if profile_rows else
    pd.DataFrame(columns=[
        "primary_segment", "attribute_label", "label_count",
        "dimension_mentioned_count", "attribute_dimension", "pct_within_dimension"
    ])
)

if not segment_profile_long.empty:
    segment_profile_long = segment_profile_long.merge(
        attribute_label_meta,
        on=["attribute_dimension", "attribute_label"],
        how="left"
    )

save_outputs(segment_profile_long, "segment_profile_long")


# =========================================================
# feature_quotes（句子级）
# =========================================================
print("写出 feature_quotes（句子级原声）...")
save_outputs(feature_quotes_df, "feature_quotes")


# =========================================================
# bundle_quotes
# =========================================================
print("写出 bundle_quotes ...")
save_outputs(bundle_quotes_df, "bundle_quotes")


# =========================================================
# noise_reviews
# =========================================================
noise_keep = ["review_id", "Asin", "Brand", "Nation", "Rating", "Content"]
noise_reviews = noise_df[noise_keep].copy()
save_outputs(noise_reviews, "noise_reviews")


print("处理完成。")
print(f"总评论数: {len(df)}")
print(f"可分群评论数: {len(classified_df)}")
print(f"噪音评论数: {len(noise_df)}")
print(f"可分群占比: {len(classified_df) / len(df):.1%}")
print("输出目录:", PROCESSED_DIR)
