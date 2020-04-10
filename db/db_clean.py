from sqlalchemy.sql import select, update
from sqlalchemy.sql.expression import func
from db.models.Corporate import Corporate
from db.models.Entity import Entity
from db.models.CorpTx import CorpTx  # noqa - needed for sqlalchemy table
from db.models.Financial import Financial  # noqa - needed for sqlalchemy table
from db.models.EquityPx import EquityPx  # noqa - needed for sqlalchemy table
from db.models.DB import db, Base


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


def del_invalid_cpn_types(table):
    # only vanilla fixed coupon bonds
    db.query(table) \
        .filter(table.cpn_type_cd != 'FXPV') \
        .delete(synchronize_session=False)
    db.commit()


def del_invalid_sub_prdct_types(table):
    # delete rows with invalid sub product type codes
    db.query(table).filter(
        table.sub_prdct_type.in_(INVALID_SUB_PRDCT_TYPES)
    ).delete(synchronize_session=False)
    db.commit()


def del_invalid_debt_types(table):
    # delete rows with invalid debt type codes
    db.query(table).filter(
        table.debt_type_cd.in_(INVALID_DEBT_TYPES)
    ).delete(synchronize_session=False)
    db.commit()


def del_invalid_scrty_ds(table):
    # only senior unsecured notes
    db.query(table).filter(
        table.scrty_ds.notin_(VALID_SCRTY_DS)
    ).delete(synchronize_session=False)
    db.commit()


def update_scrty_ds(table):
    # standardize Senior Unsecured
    db.query(table).update(
        {table.scrty_ds: 'Senior Unsecured'},
        synchronize_session=False)
    db.commit()


def update_corporate_fk():
    # corporates matching 1st 6 digits of CUSIP-9 with entity CUSIP-6
    s = select([
        Corporate.id,
        Entity.id,
        Corporate.entity_id,
        Corporate.cusip9,
        Entity.cusip6]).where(
            Corporate.entity_id.is_(None)
        ).where(
            func.left(Corporate.cusip9, 6) == Entity.cusip6
        )
    rows = db.execute(s).fetchall()

    # update entity_id for CUSIP-6 matches
    for cid, eid, _, cusip9, cusip6 in rows:
        db.query(Corporate).filter(
            Corporate.id == cid
        ).update({
            Corporate.entity_id: eid
        }, synchronize_session=False)
        db.commit()

    # corporates matching company_symbol with entity ticker
    s = select([
        Corporate.id,
        Entity.id,
        Corporate.entity_id,
        Corporate.company_symbol,
        Entity.ticker]).where(
            Corporate.entity_id.is_(None)
        ).where(
            Corporate.company_symbol == Entity.ticker
        )
    rows = db.execute(s).fetchall()

    # update entity_id for ticker matches
    for r in rows:
        db.query(Corporate).filter(
            Corporate.id == r[0]
        ).update({
            Corporate.entity_id: r[1]
        }, synchronize_session=False)
        db.commit()


def update_financial_fk():
    # financials matching ticker with entity ticker
    financial = Base.metadata.tables['financial']
    entity = Base.metadata.tables['entity']
    s = update(financial).where(
        financial.columns.entity_id.is_(None)
    ).where(
        financial.columns.ticker == entity.columns.ticker
    ).values(entity_id=entity.columns.id)
    db.execute(s)
    db.commit()


def update_equity_px_fk():
    # equity pxs matching ticker with entity ticker
    equity_px = Base.metadata.tables['equity_px']
    entity = Base.metadata.tables['entity']
    s = update(equity_px).where(
        equity_px.columns.entity_id.is_(None)
    ).where(
        equity_px.columns.ticker == entity.columns.ticker
    ).values(entity_id=entity.columns.id)
    db.execute(s)
    db.commit()


def update_corp_tx_fk():
    corp_tx = Base.metadata.tables['corp_tx']
    corporate = Base.metadata.tables['corporate']

    # corp_tx cusip_id matches corporate cusip9
    s = update(corp_tx).where(
        corp_tx.columns.corporate_id.is_(None)
    ).where(
        corp_tx.columns.cusip_id == corporate.columns.cusip9
    ).values(corporate_id=corporate.columns.id)
    db.execute(s)
    db.commit()


def del_zero_cpn(table):
    # remove zero coupon bonds
    db.query(table).filter(
        table.cpn_rt == 0
    ).delete(synchronize_session=False)
    db.commit()


def del_high_cpn(table):
    # remove high coupon bonds
    db.query(table).filter(
        table.cpn_rt >= 15
    ).delete(synchronize_session=False)
    db.commit()


def del_no_entity(table):
    db.query(table).filter(
        table.entity_id.is_(None)
    ).delete(synchronize_session=False)
    db.commit()


def del_no_corporate(table):
    db.query(table).filter(
        table.corporate_id.is_(None)
    ).delete(synchronize_session=False)
    db.commit()


def clean_db():
    # corporate
    del_invalid_cpn_types(Corporate)
    del_invalid_sub_prdct_types(Corporate)
    del_invalid_debt_types(Corporate)
    del_invalid_scrty_ds(Corporate)
    del_zero_cpn(Corporate)
    del_high_cpn(Corporate)
    update_scrty_ds(Corporate)
    update_corporate_fk()
    del_no_entity(Corporate)

    # financial
    update_financial_fk()

    # equity_px
    update_equity_px_fk()

    # corp_tx
    update_corp_tx_fk()
    del_no_corporate(CorpTx)
    del_invalid_debt_types(CorpTx)
    del_invalid_scrty_ds(CorpTx)
    del_zero_cpn(CorpTx)
    del_high_cpn(CorpTx)
    update_scrty_ds(CorpTx)


if __name__ == '__main__':
    clean_db()
