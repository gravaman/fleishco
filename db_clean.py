from models.Corporate import Corporate
from models.DB import db


INVALID_SUB_PRDCT_TYPES = ['CHRC', 'ELN', 'AGCY']
INVALID_DEBT_TYPES = [
    'S-BNT', 'SUBBNT', 'B-BNT', 'S-DEB', 'SRDEB', 'B-DEB', 'SUBDEB',
    'IDX', 'PASSTHRU', 'SSC-PTC', 'CTF', 'EQUIPTR', 'CAPSEC', 'CVDBND',
    'SSC-COV', '1RM-BND', 'LPN', 'SBN-NT', 'B-SUNT', 'B-SPV', 'COLLTR',
    'S-LPN', 'OTH-PTC', 'SC-PTC', 'JRSUBDEB', 'REFMTG', 'JSB-SPV',
    '1STRFMTG', 'B-CSEC', '1RM-NT', 'SSC-CTR', 'SC-COV', 'TRPFDSEC',
    'SC-LOB', 'C2', 'UN-CSEC', 'B-LPN', '1LN-PTC', 'RM-BND', 'INCBND',
    'UN-LPN', 'S-I4', 'S-I10', 'UN-PTC', 'JSB-DEB', 'SB-DEB', 'B-PF',
    'JSC-PTC', 'S-I11', 'M-COV', 'SECLOB', 'S-PTC', 'REFBND', 'TRCTF',
    'SC-FAC', '1LN-DEB', 'UN-DEB', 'RM-NT', 'SRSUBDEB', 'JSB-PF', 'UN-CTF',
    'ABS', 'SC-EQTR', 'SSC-EETC', 'JSB-CSEC', 'UN-SPV', 'OTH-CLUT', 'SSC-ACS',
    'S-INHB', 'B-CTF', 'UNIT', 'JSB-SUNT', 'SC-CTF', 'S-GDN', 'GENNT',
    'PFOFFEN', 'S-CTF', 'SC-CSEC', 'S-CLN', 'UN-GDN', 'SBS-PTC', 'S-EQTR',
    'SUKWAK', 'NOBAC', 'B-DRTH', 'SUKMUD', 'DEPNT', 'JSB-PTC', 'BSC-LPN',
    'STPROD', 'UN-LOB', 'SSC-DRTH', 'SUKUK', 'SSC-EQTR', 'BSC-TCT', 'M-PTC',
    '1M-CR', 'S-SUK', 'S-CSEC', 'B-DPSH', 'INCNT', 'SAKOBL', 'S-TR',
    '2LN-PTC', 'EQUNIT', 'SUBCPD', '1M-PTC', '1M-CTR', 'LSNT', 'CUSTRCPT',
    'BSC-PTC', 'S-I8', 'S-TCT', 'SSC-CLN', 'SC-DEB', 'SSC-CTF', 'SB-CSEC',
    'B-CPD', 'SC-TCT', 'B-INT', 'OTH-EQUT', 'UN-N4', 'S-FIS', 'CDO',
    'B-TR', 'S-CAT', 'JSB-LPN', 'S-SP', 'BSC-PF', 'B-PTC', 'OTH-DEB',
    'BA', 'SB-DRTH', 'SSC-C2', 'DEPSH', 'BSC-BND', 'UN-INHB', 'OTCH-CRCH',
    'BSC-DEB', 'FIS', 'PFDSTK', 'OTH-CAT', 'SB-LPN', 'UN-PRIN', 'DISNT',
    'UN-BNT', 'S-DRTH', 'SSC-CR', 'JSB-DPSH', 'BNT', 'UN-CRCH', 'SC-C2',
    'SRBNT', 'SC-CAT', 'UN-C2', 'UN-CAT', 'CAT', 'UN-INT', 'PRIN', '1M-CRCH',
    'DEB', 'B-TCT', 'JRSEC', '2LN-TGNT', 'SSC-TGNT', '2M-NT', 'SB-TGNT',
    '3RDMTG', '2M-NT', 'SB-TGNT', '3RDMTG', '2M-BND', 'S-TGNT', 'SSC-OTH',
    'JSC-NT', 'BSC-NT', 'SRSUBSEC', 'M-NT', 'MTGNT', 'MTGBND', 'SBS-NT',
    'B-OTH', '1STMTGNT', '2LN-BND', 'M-BND', 'TGNT', '1LN-BND', 'SRSEC',
    '1LN-NT', '2LN-NT', 'SECNT', '1M-NT', 'SC-BND', 'SC-NT', 'SSC-BND',
    '1M-BND', '1STMTG', 'SC-OTH', 'SSC-NT', 'SECBND', 'OTH', 'OTH-OTH',
    'OTH-BND', 'JSB-BND', 'JSB-NT', 'SB-BND', 'JRSUBNT', 'B-BND', 'SB-NT',
    'SUBNT', 'B-NT', 'SRSUBNT', 'S-OTH', 'OTH-NT', 'UN-OTH'
]
VALID_SCRTY_DS = [
    'Senior Unsecured Note', 'Senior Note', 'Unsecured Note',
    'Senior Unsecured Bond', 'Note', 'Bond', 'Sr Note', 'Unsecured Bond',
    'Unsecrd Nt'
]


def del_invalid_cpn_types():
    # only vanilla fixed coupon bonds
    db.query(Corporate).filter(
        Corporate.cpn_type_cd != 'FXPV'
    ).delete(synchronize_session=False)
    db.commit()


def del_invalid_sub_prdct_types():
    # delete rows with invalid sub product type codes
    db.query(Corporate).filter(
        Corporate.sub_prdct_type.in_(INVALID_SUB_PRDCT_TYPES)
    ).delete(synchronize_session=False)
    db.commit()


def del_invalid_debt_types():
    # delete rows with invalid debt type codes
    db.query(Corporate).filter(
        Corporate.debt_type_cd.in_(INVALID_DEBT_TYPES)
    ).delete(synchronize_session=False)
    db.commit()


def del_invalid_scrty_ds():
    # only senior unsecured notes
    db.query(Corporate).filter(
        Corporate.scrty_ds.notin_(VALID_SCRTY_DS)
    ).delete(synchronize_session=False)
    db.commit()


def clean_db():
    del_invalid_cpn_types()
    del_invalid_sub_prdct_types()
    del_invalid_debt_types()
    del_invalid_scrty_ds()
