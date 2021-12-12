'''
oracle : 기상, TOPIS
impala : 교통공사, 유동인
'''


'''테이블 정보'''
'''
oracle
TB_CI_MT_WEATHERAREA: 기상 테이블
VW_CI_IT_CURR_SPD: TOPIS 테이블 (현재 VIEW)

impala
tb_bc_ht_incident: 서울시 도로공사 정보
tb_bc_ht_incident_korea: 전국 도로공사 정보
tb_bc_ht_population: 서울시 유동인구 데이터
'''


from impala.dbapi import connect
from impala.util import as_pandas
import cx_Oracle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

road_env_data = '/Users/wbchoi/PycharmProjects/ai_projects/C_ITS/road_environments.csv'


'''IMPALA'''
impala_query = 'select * from seoul_cits.tb_bc_ht_population limit 100'

conn = connect(host='cits2.seoul.go.kr', port=21050)
impala_cursor = conn.cursor()

impala_cursor.execute(impala_query)
impala_df = as_pandas(impala_cursor)

# print(impala_cursor.description)

# print(impala_df.head(10))



'''ORACLE'''
# connection = cx_Oracle.connect("dev_user", "dev_pw", "192.168.100.187:9027/orcl", encoding = 'utf-8')
connection = cx_Oracle.connect("dev_user/dev_pw@192.168.100.187:9027/orcl")

# oracle_query = 'select * from TB_CI_MT_WEATHERAREA'
oracle_query = 'select * from VW_CI_IT_CURR_SPD'

# json
oracle_cursor = connection.cursor()
oracle_cursor.execute(oracle_query)

res = oracle_cursor.fetchall()
# print(res)

# data frame
oracle_df = pd.read_sql_query(oracle_query, connection)
print(oracle_df.head(10))
print(oracle_df.columns)


road_env = pd.read_csv(road_env_data, encoding = 'euckr')
# print(road_env.shape)

# print(road_env.columns)

road_env = road_env[[
        'LINK_ID', 'SUB_YN', 'BUS_YN', 'CW_YN', 'A3_ROADTYPE_1_YN', 'A1_LANE_04_YN', 'A1_BARR_03_YN', 'A1_BARR_02_YN',
        'SUB_CNT','S_P_UTERNX_CNT', 'SN_P_HDUFID_CNT', 'SF_P_ALL_CNT',
        'SF_PL_CW_CNT', 'SF_L_1_CNT',  'NODE_LANES_CNT', 'NODE_INTERS_CNT',
        'CW_CNT', 'BUS_CNT', 'A3_ROADTYPE_3_CNT', 'A3_ROADTYPE_1_CNT',
        'A2_STOP_2_CNT', 'A2_STOP_1_CNT', 'A1_LANE_04_CNT', 'A1_BARR_05_CNT',
         'A1_BARR_03_CNT', 'A1_BARR_02_CNT', 'ROAD_TYPE', 'REST_VEH',  'MAX_SPD', 'LANES',
        ]]


# print(road_env.shape)


scaler = MinMaxScaler()
road_env.iloc[:, 8:26] = scaler.fit_transform(road_env.iloc[:, 8:26])

road_env.iloc[:, 1:8] = np.where(road_env.iloc[:, 1:8] == 'Y', 1, 0)

# print(road_env.head())

# print(road_env['LINK_ID'].head())


