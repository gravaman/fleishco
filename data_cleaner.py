import os
import pandas as pd
import psycopg2


conn = psycopg2.connect(dbname='fleishco',
                        user='fleishco',
                        password=os.environ['PG_FLEISHCO_PASS'])
cur = conn.cursor()


def get_fund_names():
    sql_msg = 'SELECT (name) FROM funds;'
    cur.execute(sql_msg)
    return [entry[0] for entry in cur.fetchall()]


def get_fund_paths(names, base='data/q2_2019/'):
    return [base + n.replace(" ", "_").lower() + '.csv' for n in names]


def clean_funds(file_path):
    df = pd.read_csv(file_path)
    print(df)
    sql_msg = 'INSERT INTO funds (name) VALUES (%s) ON CONFLICT DO NOTHING;'
    for _, row in df.iterrows():
        print(f"adding {row['fund_name']}")
        cur.execute(sql_msg, (row['fund_name'],))
        conn.commit()


def clean_positions(fund_name, file_path, should_store=False):
    col_names = ['Stock', 'Symbol', 'Type', 'Shares Held', 'Market Value',
                 '% of Portfolio']
    dtypes = {
        'Stock': str,
        'Symbol': str,
        'Type': str,
        'Shares Held': float,
        'Market Value': float,
        '% of Portfolio': float,
    }
    df = pd.read_csv(file_path,
                     usecols=col_names,
                     dtype=dtypes,
                     )
    df.columns = ['pos_name', 'symbol', 'Type', 'shares', 'mv', 'port_pct']
    df = df[df.shares > 0]
    df = df[df.Type.isna()]
    df = df[df.symbol != 'unknown']
    print(f'positions for: {fund_name}')
    print(df)

    if should_store:
        sql_msg = f'INSERT INTO positions (fund_name, pos_name, symbol, ' \
            f'shares, mv, port_pct, source_date) ' \
            f'VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING;'
        for _, row in df.iterrows():
            print(f"adding {row.symbol}")
            vals = (fund_name, row.pos_name, row.symbol, row.shares, row.mv,
                    row.port_pct, '2019-06-30')
            cur.execute(sql_msg, vals)
            conn.commit()


def add_fund_positions():
    funds = get_fund_names()
    fps = get_fund_paths(funds)
    for (fund, fund_path) in zip(funds, fps):
        clean_positions(fund, fund_path, should_store=True)


def add_edges():
    for name in get_fund_names():
        create_edges(name)


def create_edges(fund_name):
    sql_positions = f'SELECT * FROM positions WHERE fund_name = %s ' \
        f'AND port_pct >= 1.0;'
    cur.execute(sql_positions, (fund_name,))
    positions = cur.fetchall()
    edges = []

    sql_edge = f'INSERT INTO edges (fund_name, pos0, pos1, source_date, ' \
        f' pos0_port_pct, pos1_port_pct) VALUES (%s, %s, %s, %s, %s, %s) ' \
        f'ON CONFLICT DO NOTHING;'
    source_date = '2019-06-30'
    for i, pos0 in enumerate(positions):
        e0 = pos0[0]
        pos0_name = pos0[2]
        pos0_port_pct = pos0[6]
        for j, pos1 in enumerate(positions):
            if i != j:
                e1 = pos1[0]
                pos1_name = pos1[2]
                pos1_port_pct = pos1[6]
                if (e0, e1) not in edges and (e1, e0) not in edges:
                    print(f'{fund_name, pos0_name, pos1_name, source_date}')
                    cur.execute(sql_edge, (fund_name, pos0_name,
                                           pos1_name, source_date,
                                           pos0_port_pct, pos1_port_pct))
                    conn.commit()
                    edges.append((e0, e1))


def find_pairs(threshold=3, verbose=True):
    sql_edges = f'SELECT * FROM edges;'
    cur.execute(sql_edges)
    edges = [[e[1], e[2], e[3]] for e in cur.fetchall()]
    pairs = {}
    for (fund_name, pos0, pos1) in edges:
        if pos0 in pairs:
            if pos1 in pairs[pos0]:
                pairs[pos0][pos1] += 1
            else:
                pairs[pos0][pos1] = 1
        elif pos1 in pairs:
            if pos0 in pairs[pos1]:
                pairs[pos1][pos0] += 1
            else:
                pairs[pos1][pos0] = 1
        else:
            pairs[pos0] = {pos1: 1}

    valid_pairs = []
    for pos0 in pairs:
        for pos1 in pairs[pos0]:
            if pairs[pos0][pos1] >= threshold:
                valid_pairs.append((pos0, pos1, pairs[pos0][pos1]))
    if verbose:
        print(f'pair count: {len(valid_pairs)}')

    return valid_pairs


if __name__ == '__main__':
    # clean_funds('data/funds.csv')
    find_pairs()
    conn.close()
