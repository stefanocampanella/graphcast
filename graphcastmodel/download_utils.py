MED_CMCC_PHY_REAN = "MEDSEA_MULTIYEAR_PHY_006_004"
MED_OGS_BGC_REAN = "MEDSEA_MULTIYEAR_BGC_006_008"

DATASET_IDS = ("med-ogs-bio-rean-d",
               "med-ogs-car-rean-d",
               "med-ogs-nut-rean-d",
               "med-ogs-pft-rean-d",
               "med-cmcc-cur-rean-d",
               "med-cmcc-sal-rean-d",
               "med-cmcc-tem-rean-d")

import copernicusmarine as cm
from typing import Union, Iterable

CatalogueTree = Union[dict[str, Iterable['CatalogueTree']], list[Iterable['CatalogueTree']], str]

catalogue = cm.catalogue(include_datasets=True)

def find_first_matching_subtree(tree: CatalogueTree, search_key: str, valid_values: list[str]) -> CatalogueTree:
    """Finds first subtree which is a `dict`, that has `search_key` among its keys, whose value is among `valid_values`

    Args:
        tree: tree to search for valid subtree
        search_key: key to be searched
        valid_values: list of valid value names

    Returns:
        First valid subtree
    """
    if isinstance(tree, list):
        for subtree in tree:
            find_first_matching_subtree(subtree, search_key, valid_values)
    elif isinstance(tree, dict):
        if search_key in tree.keys() and tree[search_key] in valid_values:
            return tree
        else:
            for subtree in tree.values():
                find_first_matching_subtree(subtree, search_key, valid_values)
