"""
Aesthetic definitions: each key maps to a list of text prompts that
describe the visual style. CLIP averages these into one prototype embedding.
"""

AESTHETICS = {
    "y2k": [
        "Y2K early 2000s fashion outfit",
        "low rise jeans butterfly clips metallic fabric 2000s style",
        "velour tracksuit tiny tinted sunglasses Y2K aesthetic fashion",
        "rhinestone embellished crop top flare jeans early 2000s",
        "colorful fun playful Y2K pop star inspired fashion look",
    ],
    "streetwear": [
        "streetwear urban fashion outfit on a person",
        "oversized hoodie baggy cargo pants chunky sneakers street style",
        "graphic tee joggers limited edition sneakers streetwear look",
        "urban hypebeast fashion oversized silhouette bold logo",
        "skate-inspired streetwear puffer jacket fitted cap",
    ],
    "cottagecore": [
        "cottagecore aesthetic outfit fashion",
        "floral prairie dress linen apron nature cottage garden style",
        "soft feminine countryside vintage embroidered blouse skirt",
        "romantic pastoral fashion mushroom moody forest floral",
        "whimsical vintage cottagecore woven basket floral headband",
    ],
    "dark_academia": [
        "dark academia aesthetic outfit fashion",
        "tweed blazer turtleneck sweater plaid skirt dark academia style",
        "scholarly vintage dark moody academic library fashion look",
        "brown burgundy plaid wool coat dark intellectual fashion",
        "gothic academic preppy dark muted tones blazer loafers",
    ],
    "minimalist": [
        "minimalist fashion aesthetic clean outfit",
        "neutral tones beige white black simple capsule wardrobe fashion",
        "monochrome clean lines elegant minimalist clothing look",
        "Scandinavian simple refined wardrobe neutral minimalist style",
        "effortless chic minimal silhouette no print solid color outfit",
    ],
    "preppy": [
        "preppy collegiate fashion outfit",
        "polo shirt khaki chino pants loafers preppy Ivy League style",
        "nautical stripe button-down clean cut New England preppy look",
        "argyle sweater pleated skirt headband classic preppy aesthetic",
        "country club fashion pastel cardigan loafers preppy chic",
    ],
    "boho": [
        "bohemian boho fashion outfit",
        "flowy maxi dress fringe jacket suede boots earthy boho chic",
        "free-spirit vintage layered jewelry peasant top boho aesthetic",
        "festival fashion crochet top wide-leg jeans boho hippie style",
        "earthy terracotta sunset tones boho layered natural fiber outfit",
    ],
    "grunge": [
        "grunge aesthetic fashion outfit",
        "ripped jeans band tee flannel shirt combat boots 90s grunge",
        "dark edgy alternative fashion distressed clothing grunge look",
        "vintage thrift store layered plaid grunge aesthetic outfit",
        "oversized worn-in dark alternative rocker grunge style fashion",
    ],
    "formal": [
        "formal business professional outfit",
        "tailored suit dress shirt tie classic formal business attire",
        "elegant evening wear gown cocktail dress formal fashion",
        "blazer trousers oxford shoes crisp professional formal look",
        "polished office formal wear structured blazer pencil skirt",
    ],
    "casual": [
        "casual everyday fashion outfit",
        "comfortable jeans T-shirt sneakers casual relaxed everyday wear",
        "laid-back weekend outfit hoodie sweatpants casual style",
        "simple everyday basics denim jacket white shirt casual look",
        "effortless casual chic comfortable stylish everyday fashion",
    ],
}

# Human-readable display names
AESTHETIC_NAMES = {
    "y2k": "Y2K",
    "streetwear": "Streetwear",
    "cottagecore": "Cottagecore",
    "dark_academia": "Dark Academia",
    "minimalist": "Minimalist",
    "preppy": "Preppy",
    "boho": "Boho / Bohemian",
    "grunge": "Grunge",
    "formal": "Formal",
    "casual": "Casual",
}
