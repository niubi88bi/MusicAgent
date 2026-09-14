"""普通话十三辙映射表。"""

from __future__ import annotations


RHYME_GROUPS: dict[str, frozenset[str]] = {
    "发花": frozenset({"a", "ia", "ua"}),
    "梭波": frozenset({"e", "o", "uo"}),
    "乜斜": frozenset({"ê", "ie", "üe"}),
    "一七": frozenset({"i", "ü", "er"}),
    "姑苏": frozenset({"u"}),
    "怀来": frozenset({"ai", "uai"}),
    "灰堆": frozenset({"ei", "uei"}),
    "遥条": frozenset({"ao", "iao"}),
    "由求": frozenset({"ou", "iou"}),
    "言前": frozenset({"an", "ian", "uan", "üan"}),
    "人辰": frozenset({"en", "in", "uen", "ün"}),
    "江阳": frozenset({"ang", "iang", "uang"}),
    "中东": frozenset({"eng", "ing", "ueng", "ong", "iong"}),
}

# 项目对“压大韵、押响韵”的可配置规则权重；D6 仅作为小幅加分，
# 不把它当成已经被项目验证过的听感因果结论。
LOUD_RHYME_GROUPS = frozenset({"发花", "江阳", "言前", "人辰", "中东"})

_FINAL_ALIASES = {
    "ui": "uei",
    "iu": "iou",
    "un": "uen",
    "v": "ü",
    "ve": "üe",
    "van": "üan",
    "vn": "ün",
    "ue": "üe",
}


def normalize_final(final: str) -> str:
    """统一 pypinyin 与外部输入中常见的缩写或 v/ü 写法。"""

    cleaned = final.lower().strip().replace(" ", "")
    return _FINAL_ALIASES.get(cleaned, cleaned)


def rhyme_group_for_final(final: str) -> str | None:
    """返回韵母所属十三辙；无法覆盖时返回 None 而非猜测。"""

    normalized = normalize_final(final)
    for group, finals in RHYME_GROUPS.items():
        if normalized in finals:
            return group
    return None


def normalize_rhyme_group(value: str) -> str | None:
    """把标准辙名或代表韵母统一为十三辙名称；未知值返回 None。"""

    cleaned = value.strip()
    if cleaned in RHYME_GROUPS:
        return cleaned
    if not cleaned:
        return None
    return rhyme_group_for_final(cleaned)
