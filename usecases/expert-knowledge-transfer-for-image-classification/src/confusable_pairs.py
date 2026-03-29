"""
Curated list of confusable species pairs for the KF teaching experiment.

Pairs were selected by computing cosine similarity between per-class attribute
vectors (312-dim, class_attribute_labels_continuous.txt), then filtering for
ornithological plausibility and feature-group diversity.

Each entry documents:
  - The attribute-based cosine similarity score (from CUB-200-2011 data)
  - The key visual discriminators that an expert would use
  - The field-guide source for the teaching text (to be populated under teaching_sessions/)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class ConfusablePair:
    class_id_a: int
    class_name_a: str          # plain name, e.g. "American Crow"
    class_id_b: int
    class_name_b: str
    cosine_sim: float
    feature_group: str         # primary visual dimension that discriminates
    key_discriminators: List[str]
    teaching_file: str         # filename under teaching_sessions/


# ---------------------------------------------------------------------------
# 15 curated pairs — chosen for feature-group diversity and pedagogical clarity
# ---------------------------------------------------------------------------

CONFUSABLE_PAIRS: List[ConfusablePair] = [

    # --- Corvids ---
    ConfusablePair(
        class_id_a=29, class_name_a="American Crow",
        class_id_b=30, class_name_b="Fish Crow",
        cosine_sim=0.9960,
        feature_group="size + voice",
        key_discriminators=[
            "Fish Crow is slightly smaller with a more slender bill",
            "Nasal, high-pitched 'uh-uh' call vs. Crow's lower 'caw'",
            "Fish Crow has a shorter, squarer tail in flight",
            "Wingbeat of Fish Crow is faster and more buoyant",
        ],
        teaching_file="american_crow_vs_fish_crow.md",
    ),
    ConfusablePair(
        class_id_a=107, class_name_a="Common Raven",
        class_id_b=108, class_name_b="White-necked Raven",
        cosine_sim=0.9572,
        feature_group="plumage patch + range",
        key_discriminators=[
            "White-necked Raven has a white base to neck feathers visible when ruffled",
            "White-necked Raven has a shorter, deeper bill with more curved culmen",
            "Range: White-necked is African/Middle Eastern; Common Raven is Holarctic",
            "White-necked Raven is smaller overall",
        ],
        teaching_file="common_raven_vs_white_necked_raven.md",
    ),

    # --- Shrikes ---
    ConfusablePair(
        class_id_a=111, class_name_a="Loggerhead Shrike",
        class_id_b=112, class_name_b="Great Grey Shrike",
        cosine_sim=0.9776,
        feature_group="facial mask + rump + size",
        key_discriminators=[
            "Great Grey Shrike is noticeably larger with a longer tail",
            "Loggerhead has a broader, more solid black mask extending above the bill base",
            "Great Grey shows a white rump patch in flight; Loggerhead's rump is grey",
            "Great Grey has a longer, more hooked bill",
            "Underparts: Great Grey is finely barred; Loggerhead is white to pale grey",
        ],
        teaching_file="loggerhead_shrike_vs_great_grey_shrike.md",
    ),

    # --- Buntings / Grosbeaks ---
    ConfusablePair(
        class_id_a=14, class_name_a="Indigo Bunting",
        class_id_b=54, class_name_b="Blue Grosbeak",
        cosine_sim=0.9719,
        feature_group="size + bill + wingbars",
        key_discriminators=[
            "Blue Grosbeak is much larger — almost twice the mass of Indigo Bunting",
            "Blue Grosbeak has a massive, conical bill; Indigo has a small, seed-cracker bill",
            "Blue Grosbeak shows two rusty-chestnut wingbars; Indigo lacks bold wingbars",
            "Blue Grosbeak has chestnut on the wings; Indigo is solid blue with no chestnut",
            "Female Blue Grosbeak is rich brown with wingbars; female Indigo is streaked buff-brown",
        ],
        teaching_file="indigo_bunting_vs_blue_grosbeak.md",
    ),

    # --- Gulls ---
    ConfusablePair(
        class_id_a=59, class_name_a="California Gull",
        class_id_b=62, class_name_b="Herring Gull",
        cosine_sim=0.9700,
        feature_group="bill markings + eye + mantle shade",
        key_discriminators=[
            "California Gull has both a red and black spot on the lower mandible; Herring has only red",
            "California Gull has a dark eye (brown iris); Herring Gull has a pale yellow eye",
            "California Gull is smaller and more slender with a rounder head",
            "California Gull's mantle is slightly darker grey than Herring Gull",
            "California Gull's legs are greenish-yellow; Herring Gull's legs are flesh-pink",
        ],
        teaching_file="california_gull_vs_herring_gull.md",
    ),
    ConfusablePair(
        class_id_a=62, class_name_a="Herring Gull",
        class_id_b=64, class_name_b="Ring-billed Gull",
        cosine_sim=0.9675,
        feature_group="size + bill ring + eye + legs",
        key_discriminators=[
            "Ring-billed Gull is noticeably smaller and more slender",
            "Ring-billed has a distinct black ring near the bill tip; Herring lacks this",
            "Ring-billed has yellow legs; Herring Gull has flesh-pink legs",
            "Ring-billed Gull reaches adult plumage in 3 years; Herring takes 4 years",
            "Ring-billed has a pale yellow eye with a narrow yellow orbital ring",
        ],
        teaching_file="herring_gull_vs_ring_billed_gull.md",
    ),

    # --- Terns ---
    ConfusablePair(
        class_id_a=144, class_name_a="Common Tern",
        class_id_b=146, class_name_b="Forsters Tern",
        cosine_sim=0.9678,
        feature_group="tail pattern + bill + wingtip",
        key_discriminators=[
            "Forsters Tern has silvery-white outer primaries; Common Tern's primaries are dark-tipped",
            "Forsters Tern's tail is longer and whiter; Common has grey tail with white edges",
            "In non-breeding: Forsters shows a distinctive black eye-patch; Common has a full black cap",
            "Common Tern's bill is orange-red with a black tip; Forsters is orange with less black",
            "Forsters Tern has a slightly heavier, more drooped bill",
        ],
        teaching_file="common_tern_vs_forsters_tern.md",
    ),

    # --- Cormorants ---
    ConfusablePair(
        class_id_a=24, class_name_a="Red-faced Cormorant",
        class_id_b=25, class_name_b="Pelagic Cormorant",
        cosine_sim=0.9623,
        feature_group="facial skin + gloss + size",
        key_discriminators=[
            "Red-faced Cormorant has extensive bare red-orange facial skin; Pelagic's face is dull red only at bill base",
            "Red-faced Cormorant is larger and heavier-bodied",
            "Pelagic Cormorant is the slenderest cormorant — very thin neck and tiny head",
            "Pelagic Cormorant shows a brilliant iridescent green-purple gloss in breeding; Red-faced is less iridescent",
            "Red-faced has a pale, bluish bill; Pelagic's bill is dark throughout",
        ],
        teaching_file="red_faced_cormorant_vs_pelagic_cormorant.md",
    ),

    # --- Cuckoos ---
    ConfusablePair(
        class_id_a=31, class_name_a="Black-billed Cuckoo",
        class_id_b=33, class_name_b="Yellow-billed Cuckoo",
        cosine_sim=0.9593,
        feature_group="bill color + tail spots + wing rufous",
        key_discriminators=[
            "Yellow-billed Cuckoo has a yellow lower mandible; Black-billed has an all-dark bill",
            "Yellow-billed shows large white tail spots and black undertail pattern; Black-billed has smaller spots",
            "Yellow-billed shows rufous in the wing primaries visible in flight; Black-billed shows little rufous",
            "Yellow-billed has a bold yellow orbital ring; Black-billed has red orbital ring",
        ],
        teaching_file="black_billed_cuckoo_vs_yellow_billed_cuckoo.md",
    ),

    # --- Waterthrushes ---
    ConfusablePair(
        class_id_a=183, class_name_a="Northern Waterthrush",
        class_id_b=184, class_name_b="Louisiana Waterthrush",
        cosine_sim=0.9568,
        feature_group="supercilium + throat + flank streaking",
        key_discriminators=[
            "Louisiana Waterthrush has a broader, whiter supercilium that flares behind the eye; Northern's is thinner and often yellowish",
            "Louisiana Waterthrush has an unstreaked or very lightly streaked white throat; Northern has a spotted throat",
            "Louisiana's flanks and belly are buffy-pink; Northern's underparts are yellowish with dense streaking",
            "Louisiana has a larger bill and longer legs",
            "Louisiana bobs its tail more vigorously and constantly",
        ],
        teaching_file="northern_waterthrush_vs_louisiana_waterthrush.md",
    ),

    # --- Sparrows ---
    ConfusablePair(
        class_id_a=115, class_name_a="Brewer Sparrow",
        class_id_b=117, class_name_b="Clay-colored Sparrow",
        cosine_sim=0.9578,
        feature_group="face pattern + crown",
        key_discriminators=[
            "Clay-colored Sparrow has a bold pale median crown stripe; Brewer's crown is uniformly streaked without a clear median stripe",
            "Clay-colored has a strong dark lateral crown stripe; Brewer's crown pattern is finely streaked throughout",
            "Clay-colored shows a distinct grey nape; Brewer's nape is brownish and streaked",
            "Clay-colored has a pale loral region and prominent dark moustachial stripe; Brewer's face is plainer",
            "Brewer's call is a long, insect-like buzzy trill; Clay-colored's call is a flat 'chip'",
        ],
        teaching_file="brewer_sparrow_vs_clay_colored_sparrow.md",
    ),
    ConfusablePair(
        class_id_a=116, class_name_a="Chipping Sparrow",
        class_id_b=130, class_name_b="Tree Sparrow",
        cosine_sim=0.9549,
        feature_group="breast spot + bill + cap",
        key_discriminators=[
            "Tree Sparrow has a distinct dark central breast spot on an otherwise plain breast; Chipping is unstreaked with no spot",
            "Tree Sparrow has a bicolored bill — dark on top, yellow below; Chipping's bill is all dark",
            "Tree Sparrow's rufous cap extends onto the nape; Chipping's cap is brighter and more confined",
            "Chipping Sparrow has a clear white supercilium above a black eye-line; Tree's supercilium is less contrasting",
        ],
        teaching_file="chipping_sparrow_vs_tree_sparrow.md",
    ),

    # --- Flycatchers ---
    ConfusablePair(
        class_id_a=39, class_name_a="Least Flycatcher",
        class_id_b=102, class_name_b="Western Wood Pewee",
        cosine_sim=0.9550,
        feature_group="eye ring + bill + primary projection + voice",
        key_discriminators=[
            "Least Flycatcher has a bold, complete white eye ring; Western Wood Pewee has no eye ring",
            "Least Flycatcher has a shorter bill; Western Wood Pewee has a longer, broader bill",
            "Western Wood Pewee has longer primary projection giving a more attenuated wingtip",
            "Least Flycatcher's breast is white with faint olive wash on sides; Western Wood Pewee has distinct dusky vest across breast",
            "Voice is diagnostic: Least calls 'che-BEK'; Western Wood Pewee gives a descending 'pee-eer'",
        ],
        teaching_file="least_flycatcher_vs_western_wood_pewee.md",
    ),

    # --- Cowbirds ---
    ConfusablePair(
        class_id_a=26, class_name_a="Bronzed Cowbird",
        class_id_b=27, class_name_b="Shiny Cowbird",
        cosine_sim=0.9696,
        feature_group="eye + neck ruff + iridescence",
        key_discriminators=[
            "Bronzed Cowbird has a distinctive red eye; Shiny Cowbird has a dark eye",
            "Bronzed Cowbird has a prominent neck ruff that it puffs out in display; Shiny lacks this ruff",
            "Bronzed Cowbird males show bronze-green iridescence on the body; Shiny males are blue-black throughout",
            "Bronzed Cowbird is larger and heavier-bodied",
            "Female Bronzed is dark sooty-brown; female Shiny is lighter brownish with faint streaking",
        ],
        teaching_file="bronzed_cowbird_vs_shiny_cowbird.md",
    ),

    # --- Caspian vs Elegant Tern ---
    ConfusablePair(
        class_id_a=143, class_name_a="Caspian Tern",
        class_id_b=145, class_name_b="Elegant Tern",
        cosine_sim=0.9699,
        feature_group="size + bill shape + crest",
        key_discriminators=[
            "Caspian Tern is massive — the largest tern, similar in size to a Ring-billed Gull; Elegant is medium-sized",
            "Caspian has a thick, blood-red bill; Elegant has a long, slender, drooped orange-yellow bill",
            "Elegant Tern has a long, shaggy crest that droops behind; Caspian's crest is shorter and less ragged",
            "Caspian shows dark primary tips on underwing; Elegant's underwing is paler",
            "Caspian has a deep, raspy call; Elegant has a high, scratchy tern call",
        ],
        teaching_file="caspian_tern_vs_elegant_tern.md",
    ),
]


# ---------------------------------------------------------------------------
# Convenience lookups
# ---------------------------------------------------------------------------

def get_pair(class_id_a: int, class_id_b: int) -> ConfusablePair | None:
    for p in CONFUSABLE_PAIRS:
        ids = {p.class_id_a, p.class_id_b}
        if ids == {class_id_a, class_id_b}:
            return p
    return None


def all_class_ids() -> list[int]:
    ids = set()
    for p in CONFUSABLE_PAIRS:
        ids.add(p.class_id_a)
        ids.add(p.class_id_b)
    return sorted(ids)


if __name__ == "__main__":
    print(f"{len(CONFUSABLE_PAIRS)} confusable pairs across {len(all_class_ids())} species\n")
    for i, p in enumerate(CONFUSABLE_PAIRS, 1):
        print(f"{i:2}. [{p.cosine_sim:.4f}] {p.class_name_a} vs {p.class_name_b}  ({p.feature_group})")
