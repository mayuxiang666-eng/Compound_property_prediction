# ----------------------------------------------------
# Mooney Prediction Pipeline V2.0 Path Bootstrap
# ----------------------------------------------------
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PARENT_DIR)
sys.path.extend([
    PARENT_DIR,
    os.path.join(PARENT_DIR, 'data_processing'),
    os.path.join(PARENT_DIR, 'model_training'),
    os.path.join(PARENT_DIR, 'model_analysis'),
])
# ----------------------------------------------------

# %%
import psycopg2 as psy
import pandas as pd
import os
import json

import pyodbc


# %%

M1_QUERY_COMPOUNDS = [
    "M1-T09170", "M1-T33025", "M1-B00458", "M1-S08156", "M1-T15760", "M1-T13127",
    "M1-T25045", "M1-B00460", "M1-T14885", "M1-T15899", "M1-T17771", "M1-R00218",
]


def _m1_compound_query(compound):
    compound = str(compound).strip()
    if compound.startswith("M1-"):
        return compound
    return f"M1-{compound}"


def _compound_base(compound):
    compound_query = _m1_compound_query(compound)
    return compound_query.split("-", 1)[1] if "-" in compound_query else compound_query


def connect_mms(database='HESFESFEPLANT'):  # replica data in Microsoft SQL server (Hefei/APAC)
    str_home_directory = os.path.expanduser("~")
    creds = json.load(open(str_home_directory + "/datasets/home-dataset/credentials.json"))
    conn = pyodbc.connect(
        DRIVER=creds["APAC"]["driver"],
        SERVER=creds["APAC"]["server"],
        DATABASE=database,
        UID=creds["APAC"]["uid"],
        PWD=creds["APAC"]["password"],
    )
    return conn  


def connect_datamart(datamart="mustangmaster"):  # redshift datamarts primus or mustang
    str_home_directory = os.path.expanduser("~")
    creds = json.load(
        open(str_home_directory + "/datasets/home-dataset/credentials.json")
    )
    conn = psy.connect(
        host=creds[datamart]["host"],
        port=creds[datamart]["port"],
        database=creds[datamart]["database"],
        user=creds[datamart]["user"],
        password=creds[datamart]["password"],
    )
    return conn


def get_WO_ratio(compound="", start_date='2023-01-01', end_date='2025-11-01'):
    '''get WO ratio by order ID and BatchNumber'''
    compound_query = _m1_compound_query(compound)
    compound_code_suffix = _compound_base(compound)[1:]
    
    sql = f"""
    -- 对于 compound = {compound}, SFE_database = SFEPLANT
    -- 修改：移除R1/M1相关条件，只关注WO物料
    select distinct 
    t.FMF_OrderID, 
    t.Equipment, 
    t.BatchNumber, 
    t.BatchEndTime, 
    t.FMF_Recipe, 
    t.FMF_weight, 
    t.MaterialName, 
    t.MaterialCode, 
    t.ChargeID as source_WO_OrderID, 
    t.avg_rework as CM_weight, 
    t.CM_specific_ratio, 
    t_.sum_WO as sum_all_CM, 
    t_.WO_ratio as CM_ratio
    from
    (
        (select 
        batch_info.FMF_OrderID, 
        batch_info.Equipment, 
        batch_info.BatchNumber, 
        batch_info.BatchEndTime, 
        batch_info.FMF_Recipe, 
        avg(BatchWeight) as FMF_weight,
        batch_weight.MaterialName, 
        batch_weight.ChargeID, 
        batch_weight.MaterialCode,
        avg(batch_weight.FMF_rework_plus_workoff_weight) as avg_rework, 
        cast(avg(batch_weight.FMF_rework_plus_workoff_weight) as decimal)/(cast(avg(BatchWeight) as decimal))*100 as CM_specific_ratio
    from
    (
        select distinct 
            o.OrderID as FMF_OrderID, 
            bh.BatchNumber, 
            bh.BatchEndTime, 
            o.Equipment, 
            avg(bh.BatchWeight) as BatchWeight, 
            o.CompoundDescription as FMF_Recipe
        from HESFESFEPLANT.dbo.Orders o 
        join HESFESFEPLANT.dbo.RecipeHeaders rh on rh.RecipeID=o.CompoundDescription
        join HESFESFEPLANT.dbo.BatchHeader as bh on o.OrderID = bh.OrderID
        where (o.CompoundName like '{compound_query}%')
        and rh.Division='P'
        group by o.OrderID, o.CompoundDescription, bh.BatchNumber, o.Equipment, bh.BatchEndTime
    ) as batch_info
    left join 
    (
        select distinct 
            o.OrderID as FMF_OrderID, 
            rm.BatchNumber, 
            rm.MaterialName, 
            rm.ChargeID, 
            rm.MaterialCode, 
            o.CompoundDescription as FMF_Recipe, 
            avg(rm.ActualWeight) as FMF_rework_plus_workoff_weight
        from HESFESFEPLANT.dbo.Orders o 
        join HESFESFEPLANT.dbo.BatchMaterials rm on o.OrderID= rm.OrderID
        where (o.CompoundName like '{compound_query}%')
        and (rm.MaterialCode like 'CM%' 
            or rm.MaterialName like '%WO%' 
            or rm.MaterialName like '%R T01139%' 
            or rm.MaterialName like '%R T17771%' 
            or rm.MaterialName like '%R T00011%' 
            or rm.MaterialName like '%-T01139%' 
            or rm.MaterialName like '%-T17771%' 
            or rm.MaterialName like '%-T00011%' 
            or rm.MaterialCode like '%5{compound_code_suffix}%' 
            or rm.MaterialCode like '%6{compound_code_suffix}%')
        group by o.OrderID, o.CompoundDescription, rm.BatchNumber, rm.MaterialName, rm.MaterialCode, rm.ChargeID
    ) as batch_weight on (batch_weight.FMF_OrderID = batch_info.FMF_OrderID and batch_weight.BatchNumber = batch_info.BatchNumber)
        group by batch_info.FMF_OrderID, batch_info.FMF_Recipe, batch_info.BatchEndTime, batch_weight.ChargeID, 
                batch_weight.MaterialCode, batch_weight.MaterialName, batch_info.Equipment, batch_info.BatchNumber
        ) as t
    left join 
        (
            select 
            t_.FMF_OrderID, 
            t_.BatchNumber, 
            t_.FMF_Recipe, 
            cast(sum(t_.avg_rework) as float) as sum_WO,
            cast(sum(t_.avg_rework) as float)/cast(avg(t_.FMF_weight) as float)*100 as WO_ratio
        from 
        (
            select 
                batch_info.FMF_OrderID, 
                batch_info.BatchNumber, 
                batch_info.FMF_Recipe, 
                avg(batch_info.BatchWeight) as FMF_weight, 
                batch_weight.MaterialName, 
                batch_weight.ChargeID, 
                batch_weight.MaterialCode, 
                avg(batch_weight.FMF_rework_plus_workoff_weight) as avg_rework
            from
            (
                select distinct 
                    o.OrderID as FMF_OrderID, 
                    bh.BatchNumber, 
                    avg(bh.BatchWeight) as BatchWeight, 
                    o.CompoundDescription as FMF_Recipe
                from HESFESFEPLANT.dbo.Orders o 
                join HESFESFEPLANT.dbo.RecipeHeaders rh on rh.RecipeID=o.CompoundDescription
                join HESFESFEPLANT.dbo.BatchHeader as bh on o.OrderID = bh.OrderID
                where (o.CompoundName like '{compound_query}%')
                and rh.Division='P'
                group by o.OrderID, o.CompoundDescription, bh.BatchNumber
            ) as batch_info
            left join 
            (
                select distinct 
                    o.OrderID as FMF_OrderID, 
                    rm.BatchNumber, 
                    rm.MaterialName, 
                    rm.ChargeID, 
                    rm.MaterialCode, 
                    o.CompoundDescription as FMF_Recipe,  
                    avg(rm.ActualWeight) as FMF_rework_plus_workoff_weight
                from HESFESFEPLANT.dbo.Orders o 
                join HESFESFEPLANT.dbo.BatchMaterials rm on o.OrderID= rm.OrderID
                where (o.CompoundName like '{compound_query}%')
                and (rm.MaterialCode like 'CM%' 
                    or rm.MaterialName like '%WO%' 
                    or rm.MaterialName like '%R T01139%' 
                    or rm.MaterialName like '%R T17771%' 
                    or rm.MaterialName like '%R T00011%' 
                    or rm.MaterialName like '%-T01139%' 
                    or rm.MaterialName like '%-T17771%' 
                    or rm.MaterialName like '%-T00011%' 
                    or rm.MaterialCode like '%5{compound_code_suffix}%' 
                    or rm.MaterialCode like '%6{compound_code_suffix}%')
                group by o.OrderID, o.CompoundDescription, rm.BatchNumber, rm.MaterialName, rm.ChargeID, rm.MaterialCode
            ) as batch_weight on (batch_weight.FMF_OrderID = batch_info.FMF_OrderID and batch_weight.BatchNumber = batch_info.BatchNumber)
            group by batch_info.FMF_OrderID, batch_info.FMF_Recipe, batch_info.BatchNumber, batch_weight.MaterialName, 
                     batch_weight.ChargeID, batch_weight.MaterialCode
        ) as t_ 
        group by t_.FMF_OrderID, t_.BatchNumber, t_.FMF_Recipe
    ) as t_ on (t.FMF_OrderID=t_.FMF_OrderID and t.BatchNumber=t_.BatchNumber)
    )
    """
    WO_ratio = pd.read_sql(sql, connect_mms(database='HESFESFEPLANT'))

    # merge Hefei daily weather by production date (BatchEndTime)
    if len(WO_ratio) > 0 and 'BatchEndTime' in WO_ratio.columns:
        try:
            weather = get_hefei_weather(WO_ratio['BatchEndTime'])
            WO_ratio['_date_'] = pd.to_datetime(WO_ratio['BatchEndTime']).dt.date
            WO_ratio = WO_ratio.merge(weather, left_on='_date_', right_on='date', how='left')
            WO_ratio.drop(columns=['_date_', 'date'], inplace=True, errors='ignore')
        except Exception as e:
            print(f"Warning: weather merge failed: {e}")

    WO_ratio.to_csv("WO_ratio_s156.csv", index=False)

    return WO_ratio


def get_batch_curves(compound, start_date='2023-01-01', end_date='2025-11-01'):
    """Aggregate mixing curve data (Temperature, Power, Torque, RotorSpeed, WayofRam)
    per OrderID+BatchNumber directly in SQL to avoid transferring raw time-series rows.
    Curve columns (binary in SQL Server) are CAST to FLOAT inside the query."""

    compound_query = _m1_compound_query(compound)

    sql = f"""
        SELECT
            bc.OrderID,
            bc.BatchNumber,
            AVG(TRY_CAST(bc.Curve1 AS FLOAT)) AS curve_Temperature_mean,
            MAX(TRY_CAST(bc.Curve1 AS FLOAT)) AS curve_Temperature_max,
            AVG(TRY_CAST(bc.Curve2 AS FLOAT)) AS curve_Power_mean,
            AVG(TRY_CAST(bc.Curve5 AS FLOAT)) AS curve_Torque_mean,
            AVG(TRY_CAST(bc.Curve6 AS FLOAT)) AS curve_RotorSpeed_mean,
            AVG(TRY_CAST(bc.Curve7 AS FLOAT)) AS curve_WayofRam_mean
        FROM HESFESFEPLANT.dbo.BatchCurve bc
        JOIN HESFESFEPLANT.dbo.Orders o ON bc.OrderID = o.OrderID
        WHERE o.CompoundName LIKE '{compound_query}%'
          AND o.OrderStartTime >= '{start_date}'
        GROUP BY bc.OrderID, bc.BatchNumber
    """
    batch_curves = pd.read_sql(sql, connect_mms())
    return batch_curves


def get_hefei_weather(dates):
    """get daily temperature and humidity for Hefei Shushan district
    via Open-Meteo historical archive API (free, no key required).
    dates: iterable of date strings or datetime-like objects (production dates).
    Returns a DataFrame with date, temp_mean, temp_max, temp_min, humidity_mean columns."""
    import urllib.request

    dates_parsed = pd.Series(pd.to_datetime(dates, errors='coerce')).dt.date
    start = str(dates_parsed.min())
    end = str(dates_parsed.max())

    # Hefei Shushan district coordinates
    lat, lon = 31.86, 117.27

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
        f",relative_humidity_2m_mean,relative_humidity_2m_max,relative_humidity_2m_min"
        f"&timezone=Asia%2FShanghai"
    )

    with urllib.request.urlopen(url, timeout=30) as resp:
        import json as _json
        payload = _json.loads(resp.read().decode())

    daily = payload["daily"]
    weather_df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).date,
        "env_temp_mean":     daily["temperature_2m_mean"],
        "env_temp_max":      daily["temperature_2m_max"],
        "env_temp_min":      daily["temperature_2m_min"],
        "env_humidity_mean": daily["relative_humidity_2m_mean"],
        "env_humidity_max":  daily["relative_humidity_2m_max"],
        "env_humidity_min":  daily["relative_humidity_2m_min"],
    })
    return weather_df


def MMS_data(
    plant_id='',
    all_plants=False,
    MMS_data_type="WO_ratio",
    remill_included = pd.DataFrame(),
    start_date="2023-01-01",
    end_date="2025-01-01",
    compounds=None,
    #LabExcellence=False,
):
    """select the plants you want to include in the list by uncommenting the respective one"""

    plants_list = [
        ["Hefei", "HESFESFEPLANT", 0, "APAC", "HE", M1_QUERY_COMPOUNDS],
    ]
    if MMS_data_type=="WO_ratio":
        mms_data = []

        # just a single plant
        plants_list_ = next((x for x in plants_list if plant_id in x), [])
        Plant = plants_list_[0]
        print(Plant)
        Plant_SHORT_CODE = plants_list_[4]
        target_compounds = compounds if compounds is not None else plants_list_[5]
        print('start_date',start_date)
        print('end_date',end_date)

        for comp in target_compounds:
            df = get_WO_ratio(
                    compound=comp,
                    start_date=start_date,
                    end_date=end_date,
                    #plant_id=Plant_SHORT_CODE,
                )

            print(Plant + "_relevant_data_retrieved rows: ", len(df))
            df["Compound"]=comp
            df["Plant"] = Plant
            df["plant_short_code"] = Plant_SHORT_CODE
            mms_data.append(df)

    elif MMS_data_type=="mixing_traceability":
        mms_data = []

        # just a single plant
        plants_list_ = next((x for x in plants_list if plant_id in x), [])
        Plant = plants_list_[0]
        Plant_SHORT_CODE = plants_list_[4]
        target_compounds = compounds if compounds is not None else plants_list_[5]
        for comp in target_compounds:
            remill_subset = remill_included.loc[(remill_included.Compound==comp) & (remill_included['plant_short_code']==plants_list_[4])]
            if (remill_included['Compound'].isin([comp,plants_list_[4]]).any() 
                and remill_included['plant_short_code'].isin([comp,plants_list_[4]]).any() 
                and not remill_subset.empty and remill_subset['nr_remill_batches'].values[0] > 300):

                df = mixing_traceability_M1_R1_FMF(
                        compound=comp
                        #plant_id=Plant_SHORT_CODE,
                        #start_date=start_date,
                        #end_date=end_date,
                    )

                print(Plant + "_relevant_data_retrieved rows: ", len(df))
                df["Compound"]=comp
                df["Plant"] = Plant
                df["plant_short_code"] = Plant_SHORT_CODE
                mms_data.append(df)

            else:

                df = mixing_traceability_M1_FMF(
                        compound=comp
                        #plant_id=Plant_SHORT_CODE,
                        #start_date=start_date,
                        #end_date=end_date,
                    )

                print(Plant + "_relevant_data_retrieved rows: ", len(df))
                df["Compound"]=comp
                df["Plant"] = Plant
                df["plant_short_code"] = Plant_SHORT_CODE
                mms_data.append(df)    

    elif MMS_data_type=="WO_traceability":
        mms_data = []

        # just a single plant
        plants_list_ = next((x for x in plants_list if plant_id in x), [])
        Plant = plants_list_[0]
        Plant_SHORT_CODE = plants_list_[4]
        target_compounds = compounds if compounds is not None else plants_list_[5]
        for comp in target_compounds:

            df = WO_traceability(
                    compound=comp
                    #plant_id=Plant_SHORT_CODE,
                    #start_date=start_date,
                    #end_date=end_date,
                )

            print(Plant + "_relevant_data_retrieved rows: ", len(df))
            df["Compound"]=comp
            df["Plant"] = Plant
            df["plant_short_code"] = Plant_SHORT_CODE
            mms_data.append(df)

    elif MMS_data_type=="sample_batch":
        mms_data = []
        
        # just a single plant

        plants_list_ = next((x for x in plants_list if plant_id in x), [])
        Plant = plants_list_[0]
        print(Plant)
        Plant_SHORT_CODE = plants_list_[4]
        target_compounds = compounds if compounds is not None else plants_list_[5]
        for comp in target_compounds:
            df = get_sample_batch_120sampling(
                        compound=comp
                )

            print(Plant + "_relevant_data_retrieved rows: ", len(df))
            df["Compound"]=comp
            df["Plant"] = Plant
            df["plant_short_code"] = Plant_SHORT_CODE
            mms_data.append(df)

    elif MMS_data_type=="out_of_tolerance":
        mms_data = []

        # just a single plant
        plants_list_ = next((x for x in plants_list if plant_id in x), [])
        Plant = plants_list_[0]
        print(Plant)
        Plant_SHORT_CODE = plants_list_[4]
        target_compounds = compounds if compounds is not None else plants_list_[5]
        for comp in target_compounds:

            df = get_weighings_out_tolerance(
                    compound=comp
                    #plant_id=Plant_SHORT_CODE,
                    #start_date=start_date,
                    #end_date=end_date,
                )

            print(Plant + "_relevant_data_retrieved rows: ", len(df))
            df["Compound"]=comp
            df["Plant"] = Plant
            df["plant_short_code"] = Plant_SHORT_CODE
            mms_data.append(df)

    elif MMS_data_type=="master_recipe_phr":
        mms_data = []

        # just a single plant
        plants_list_ = next((x for x in plants_list if plant_id in x), [])
        Plant = plants_list_[0]
        print(Plant)
        Plant_SHORT_CODE = plants_list_[4]
        target_compounds = compounds if compounds is not None else plants_list_[5]
        for comp in target_compounds:

            df = master_recipe_phr(
                    compound=comp
                    #plant_id=Plant_SHORT_CODE,
                    #start_date=start_date,
                    #end_date=end_date,
                )

            print(Plant + "_relevant_data_retrieved rows: ", len(df))
            df["Compound"]=comp
            df["Plant"] = Plant
            df["plant_short_code"] = Plant_SHORT_CODE
            mms_data.append(df)            

    elif MMS_data_type=="master_polymer_MMS_lotID":
        mms_data = []

        # just a single plant
        plants_list_ = next((x for x in plants_list if plant_id in x), [])
        Plant = plants_list_[0]
        print(Plant)
        Plant_SHORT_CODE = plants_list_[4]
        target_compounds = compounds if compounds is not None else plants_list_[5]
        for comp in target_compounds:

            df = polymer_master_MMS_lotIDs(
                    compound=comp
                    #plant_id=Plant_SHORT_CODE,
                    #start_date=start_date,
                    #end_date=end_date,
                )

            print(Plant + "_relevant_data_retrieved rows: ", len(df))
            df["Compound"]=comp
            df["Plant"] = Plant
            df["plant_short_code"] = Plant_SHORT_CODE
            mms_data.append(df) 

    return pd.concat(mms_data, ignore_index=True)

def get_silanization_dry_mixing(step, plant, compound, start_date='2023-01-01', end_date='2025-11-01'):
    """get silanization, time to plateau and dry mixing durations and energies on M1.
    Note: For Hefei, 'plant' should be passed as 'he' (lowercase) because Redshift 
    replicated schemas use 'he' suffix instead of 'hf'."""

    conn = connect_datamart(datamart="primusmaster")

    sql = f"""
        select p.batch_information_fk, p.time_to_sil_plateau_duration,  p.top_mixer_last_step_time, p.bottom_end_time, 
		(top_mixer_last_step_time-top_mixer_start_sil_time) as top_sil_duration, (bottom_end_time-top_mixer_last_step_time) as bottom_sil_duration, 
		p.silanization_energy_MJ, p.top_avg_sil_temperature, bottom_avg_sil_temperature, 
		dry_mixing_nr_rotations, dry_mixing_duration, dry_mixing_energy_MJ
        from
        (select tbl1.batch_information_fk, time_to_sil_plateau_duration,
                top_mixer_start_sil_time, top_mixer_last_step_time,  bottom_end_time, silanization_energy_MJ,
                avg(tbl2.temperature) as top_avg_sil_temperature
        from
        (select distinct batch.batch_information_pk as batch_information_fk,
        avg(plateau.duration) as time_to_sil_plateau_duration, 
        avg(sil.sil_start_plateau_time) as top_mixer_start_sil_time, avg(last_top_time) as top_mixer_last_step_time,   
        avg(start_time_at_bottom_silanization) as start_time_at_bottom_silanization
        from primusmaster."13_production_mixing_bots"."13_02_02_mixing_batch_info_""" +plant+ """" as batch
        join primusmaster."13_production_mixing_bots"."13_02_01_mixing_order_info_""" +plant+ """" as orders on batch.order_id = orders.order_id
        left join primusmaster."13_production_mixing_bots"."13_03_02_mixing_kpi_timetoplateau_""" +plant+ """" as plateau on plateau.batch_information_fk=batch.batch_information_pk 
        left join primusmaster."05_stg"."05_02_mixing_silanization_""" +plant+ """" as sil on batch.batch_information_pk=sil.batch_information_fk     
        where (orders.compound_name_long like '"""+step+"""-"""+compound+"""%')
        and orders.order_start_time_utc >= '2023-03-01 00:00:00'
        and (plateau.step_no=4 or plateau.step_no is null) --assuming step_no is 4 or null
        group by batch.batch_information_pk
        ) as tbl1
        join
        (select distinct batch_information_fk, temperature, "time" as time1
        from primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_""" +plant+ """"
        ) as tbl2 on (tbl1.batch_information_fk=tbl2.batch_information_fk and tbl2.time1 > tbl1.top_mixer_start_sil_time and tbl2.time1 < tbl1.top_mixer_last_step_time)
        join
        (select distinct batch_information_fk, max("time") as bottom_end_time
        from primusmaster."05_stg"."05_02_mixing_silanization_""" +plant+ """"
        where "power" > 50 
        group by batch_information_fk
        ) as tbl4 on (tbl1.batch_information_fk=tbl4.batch_information_fk) 
        join
        (select distinct batch_information_fk, sum("power")/1000 as silanization_energy_MJ
        from primusmaster."05_stg"."05_02_mixing_silanization_""" +plant+ """"
        where "time" < (select max("time")
        from primusmaster."05_stg"."05_02_mixing_silanization_""" +plant+ """"
        where "power" > 50) 
        group by batch_information_fk
        ) as tbl5 on (tbl1.batch_information_fk=tbl5.batch_information_fk)     
        group by tbl1.batch_information_fk, time_to_sil_plateau_duration,
                top_mixer_start_sil_time, top_mixer_last_step_time,  bottom_end_time, silanization_energy_MJ
        ) as p 
        join
        (select tbl2.batch_information_fk, avg(tbl2.temperature) as bottom_avg_sil_temperature
        from
        (select distinct batch_information_fk, temperature, "time" as time1
        from primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_""" +plant+ """"
        where "time" < (select  max("time")
        from primusmaster."05_stg"."05_02_mixing_silanization_""" +plant+ """"
        where "power" > 50)
        and "time" > (select avg(last_top_time)
        from primusmaster."05_stg"."05_02_mixing_silanization_""" +plant+ """") 
        ) as tbl2   
        group by tbl2.batch_information_fk
        ) as q on q.batch_information_fk=p.batch_information_fk
		join (
 		select tbl1.batch_information_fk, dry_mixing_nr_rotations, dry_mixing_duration, 
                dry_mixing_start_time,dry_mixing_end_time,
                sum(tbl3."power")/1000 as dry_mixing_energy_MJ
        from
        (select distinct batch.batch_information_pk as batch_information_fk,
        avg(dry.rotation_count) as dry_mixing_nr_rotations, avg(dry.duration) as dry_mixing_duration, 
        avg(dry.start_time) as dry_mixing_start_time, avg(dry.end_time) as dry_mixing_end_time
        from primusmaster."13_production_mixing_bots"."13_02_02_mixing_batch_info_""" +plant+ """" as batch
        join primusmaster."13_production_mixing_bots"."13_02_01_mixing_order_info_""" +plant+ """" as orders on batch.order_id = orders.order_id       
        left join primusmaster."13_production_mixing_bots"."v_mixing_kpi_dry_mixing_main_""" +plant+ """" as dry on dry.batch_information_fk=batch.batch_information_pk
        where (orders.compound_name_long like '"""+step+"""-"""+compound+"""%')
        and orders.order_start_time_utc >= '2023-03-01 00:00:00'
        group by batch.batch_information_pk
        ) as tbl1
        join
        (select distinct batch_information_fk, "power", "time"
        from primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_""" +plant+ """"
        ) as tbl3 on (tbl1.batch_information_fk=tbl3.batch_information_fk and tbl3."time" > tbl1.dry_mixing_start_time and tbl3."time" < tbl1.dry_mixing_end_time)     
        group by tbl1.batch_information_fk, dry_mixing_nr_rotations, dry_mixing_duration,  dry_mixing_start_time, dry_mixing_end_time
        ) as t on t.batch_information_fk=p.batch_information_fk
            """
    
    cur = conn.cursor()
    cur.execute(sql)
    data = pd.DataFrame(
        cur.fetchall(), columns=['batch_information_fk', 'time_to_sil_plateau_duration',  'top_mixer_last_step_time', 'bottom_end_time', 
		'top_sil_duration', 'bottom_sil_duration', 'silanization_energy_MJ', 'top_avg_sil_temperature', 'bottom_avg_sil_temperature', 
		'dry_mixing_nr_rotations', 'dry_mixing_duration', 'dry_mixing_energy_MJ'])
    cur.close()
    conn.close()
    return data


def get_sfe_bot_discharge_temperature(compound, start_date='2024-01-01', end_date='2025-11-01'):
    """Get bottom discharge temperature from SFE BatchData at AVR_MSB step 4.
    This replaces the unreliable Primus-derived bottom temperature while keeping
    the same OrderID+BatchNumber join key used downstream."""

    compound_query = _m1_compound_query(compound)

    sql = f"""
        WITH ranked AS (
            SELECT
                bd.OrderID,
                bd.BatchNumber,
                TRY_CAST(bd.Value AS FLOAT) AS bot_temp_discharge_sfe,
                bd.[timestamp] AS bot_temp_timestamp_sfe,
                ROW_NUMBER() OVER (
                    PARTITION BY bd.OrderID, bd.BatchNumber
                    ORDER BY bd.[timestamp] DESC
                ) AS rn
            FROM HESFESFEPLANT.dbo.BatchData bd
            JOIN HESFESFEPLANT.dbo.Orders o
                ON bd.OrderID = o.OrderID
            WHERE o.CompoundName LIKE '{compound_query}%'
              AND o.OrderStartTime >= '{start_date}'
              AND o.OrderStartTime < '{end_date}'
              AND bd.GroupName = 'AVR_MSB'
              AND bd.StepNo = 4
              AND bd.VariablePath = 'SCP-3-Temperature-C-F'
        )
        SELECT
            OrderID,
            BatchNumber,
            bot_temp_discharge_sfe,
            bot_temp_timestamp_sfe
        FROM ranked
        WHERE rn = 1
    """
    data = pd.read_sql(sql, connect_mms())
    if len(data) > 0:
        data['OrderID'] = data['OrderID'].astype(str)
        data['BatchNumber'] = pd.to_numeric(data['BatchNumber'], errors='coerce').astype('Int64')
    return data

def get_primusdata(step, plant, compound, start_date='2024-01-01', end_date='2025-11-01'):
    """get some process parameters from relevant mixing curves such as 
    temperature at discharge, batch duration, rotor power consumption"""

    compound_query = _m1_compound_query(compound)

    conn = connect_datamart(datamart="primusmaster")

    sql = f"""
        WITH
        relevant_orders AS (
            SELECT DISTINCT
                o.order_id,
                o.compound_description,
                o.compound_name_long,
                o.prod_variant,
                o.prod_issue,
                o.prod_version
            FROM primusmaster."13_production_mixing_bots"."13_02_01_mixing_order_info_{plant}" o
            WHERE o.compound_name_long LIKE '{compound_query}%'
              AND o.order_start_time_utc >= '{start_date}'
              AND o.order_start_time_utc < '{end_date}'
        ),
        relevant_batches AS (
            SELECT DISTINCT c.batch_information_fk, c.order_id
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_orders ro
                ON c.order_id = ro.order_id
        ),
        -- top step10结束时间
        step_times_top AS (
            SELECT batch_information_fk,
                MAX(CASE WHEN step_no = 10 THEN value END) AS step10_time
            FROM primusmaster."13_production_mixing_bots"."13_02_08_mixing_batch_data_info_step_time_{plant}"
            WHERE batch_information_fk IN (SELECT batch_information_fk FROM relevant_batches)
              AND group_name = 'AVR_MST' AND step_no = 10
            GROUP BY batch_information_fk
        ),
        -- top step6/step8时间
        step_times_7_8 AS (
            SELECT batch_information_fk,
                MAX(CASE WHEN step_no = 6 THEN value END) AS step6_time,
                MAX(CASE WHEN step_no = 8 THEN value END) AS step8_time
            FROM primusmaster."13_production_mixing_bots"."13_02_08_mixing_batch_data_info_step_time_{plant}"
            WHERE batch_information_fk IN (SELECT batch_information_fk FROM relevant_batches)
              AND group_name = 'AVR_MST' AND step_no IN (6, 8)
            GROUP BY batch_information_fk
        ),
        -- step7-8 平均转速（step6_time 到 step8_time）
        rotor_speed_7_8 AS (
            SELECT c.batch_information_fk, AVG(c.rotor_speed) AS avg_rotor_speed_step7_8
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN step_times_7_8 s78 ON c.batch_information_fk = s78.batch_information_fk
                AND c."time" >= s78.step6_time
                AND c."time" <= s78.step8_time
            GROUP BY c.batch_information_fk
        ),
        -- cooling时间
        cooling_times AS (
            SELECT batch_information_fk,
                MAX(CASE WHEN step_no = 8 THEN value END) AS cooling_start_time,
                MAX(CASE WHEN step_no = 9 THEN value END) AS cooling_end_time
            FROM primusmaster."13_production_mixing_bots"."13_02_08_mixing_batch_data_info_step_time_{plant}"
            WHERE batch_information_fk IN (SELECT batch_information_fk FROM relevant_batches)
              AND group_name = 'AVR_MST' AND step_no IN (8, 9)
            GROUP BY batch_information_fk
        ),
        -- top discharge
        curves_top_discharge AS (
            SELECT DISTINCT c.batch_information_fk, c.order_id,
                c.temperature AS top_temp_discharge,
                c."time" AS top_time_discharge,
                ro.compound_description, ro.compound_name_long,
                ro.prod_variant, ro.prod_issue, ro.prod_version
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb
                ON c.batch_information_fk = rb.batch_information_fk
            JOIN relevant_orders ro
                ON c.order_id = ro.order_id
            JOIN step_times_top st ON c.batch_information_fk = st.batch_information_fk
                AND c."time" = st.step10_time
        ),
        -- top能量（time < step10_time）
        pow_top AS (
            SELECT c.batch_information_fk, c.order_id, SUM(c."power") AS power_
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN step_times_top st ON c.batch_information_fk = st.batch_information_fk
                AND c."time" < st.step10_time
            GROUP BY c.batch_information_fk, c.order_id
        ),
        -- bot AVR_MSB时间窗口（step1开始到step4结束）
        step_times_bot_window AS (
            SELECT batch_information_fk,
                MIN(value) AS bot_start_time,
                MAX(value) AS bot_end_time
            FROM primusmaster."13_production_mixing_bots"."13_02_08_mixing_batch_data_info_step_time_{plant}"
            WHERE batch_information_fk IN (SELECT batch_information_fk FROM relevant_batches)
              AND group_name = 'AVR_MSB'
            GROUP BY batch_information_fk
        ),
        -- bot能量取AVR_MSB最后一个step的能量值，而不是窗口内逐点累加
        bot_last_power_time AS (
            SELECT c.batch_information_fk, MAX(c."time") AS last_time
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN step_times_bot_window bw ON c.batch_information_fk = bw.batch_information_fk
                AND c."time" >= bw.bot_start_time
                AND c."time" <= bw.bot_end_time
            GROUP BY c.batch_information_fk
        ),
        pow_bot AS (
            SELECT c.batch_information_fk, c.order_id, c."power" AS power_
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN bot_last_power_time blp ON c.batch_information_fk = blp.batch_information_fk
                AND c."time" = blp.last_time
        ),
        -- bot discharge直接取该batch曲线的最后一个时间点
        bot_max_time AS (
            SELECT c.batch_information_fk, MAX(c."time") AS max_time
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            GROUP BY c.batch_information_fk
        ),
        bot_discharge AS (
            SELECT DISTINCT c.batch_information_fk,
                c.temperature AS bot_temp_discharge,
                c."time" AS bot_time_discharge
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN bot_max_time bmt ON c.batch_information_fk = bmt.batch_information_fk
                AND c."time" = bmt.max_time
        ),
        -- top step8结束最近时间点（<=step8_time）
        step8_max_time AS (
            SELECT c.batch_information_fk, MAX(c."time") AS max_time
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN step_times_7_8 s78 ON c.batch_information_fk = s78.batch_information_fk
                AND c."time" <= s78.step8_time
            GROUP BY c.batch_information_fk
        ),
        step8_end_temp AS (
            SELECT DISTINCT c.batch_information_fk, c.temperature AS top_step8_end_temp
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN step8_max_time s8mt ON c.batch_information_fk = s8mt.batch_information_fk
                AND c."time" = s8mt.max_time
        ),
        -- cooling start温度（取<=cooling_start_time的最近点）
        cool_start_max_time AS (
            SELECT c.batch_information_fk, MAX(c."time") AS max_time
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN cooling_times ct ON c.batch_information_fk = ct.batch_information_fk
                AND c."time" <= ct.cooling_start_time
            GROUP BY c.batch_information_fk
        ),
        cool_temp_start AS (
            SELECT DISTINCT c.batch_information_fk, c.temperature AS cooling_start_temp
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN cool_start_max_time csmt ON c.batch_information_fk = csmt.batch_information_fk
                AND c."time" = csmt.max_time
        ),
        -- cooling end温度（取<=cooling_end_time的最近点）
        cool_end_max_time AS (
            SELECT c.batch_information_fk, MAX(c."time") AS max_time
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN cooling_times ct ON c.batch_information_fk = ct.batch_information_fk
                AND c."time" <= ct.cooling_end_time
            GROUP BY c.batch_information_fk
        ),
        cool_temp_end AS (
            SELECT DISTINCT c.batch_information_fk, c.temperature AS cooling_end_temp
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN cool_end_max_time cemt ON c.batch_information_fk = cemt.batch_information_fk
                AND c."time" = cemt.max_time
        ),
        -- cooling阶段能量
        cool_energy AS (
            SELECT c.batch_information_fk, c.order_id, SUM(c."power") AS power_
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN cooling_times ct ON c.batch_information_fk = ct.batch_information_fk
                AND c."time" >= ct.cooling_start_time
                AND c."time" <= ct.cooling_end_time
            GROUP BY c.batch_information_fk, c.order_id
        ),
        -- dry mixing = step 3（step2_time → step3_time）
        step_times_dry_mixing AS (
            SELECT batch_information_fk,
                MAX(CASE WHEN step_no = 2 THEN value END) AS dm_start_time,
                MAX(CASE WHEN step_no = 3 THEN value END) AS dm_end_time
            FROM primusmaster."13_production_mixing_bots"."13_02_08_mixing_batch_data_info_step_time_{plant}"
                        WHERE batch_information_fk IN (SELECT batch_information_fk FROM relevant_batches)
                            AND group_name = 'AVR_MST' AND step_no IN (2, 3)
            GROUP BY batch_information_fk
        ),
        -- dry mixing: duration, energy, avg power
        dry_mixing_stats AS (
            SELECT c.batch_information_fk,
                dm.dm_end_time - dm.dm_start_time AS dry_mixing_duration,
                SUM(c."power") / 1000 AS dry_mixing_energy_MJ,
                AVG(c."power") AS dry_mixing_avg_power
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN step_times_dry_mixing dm ON c.batch_information_fk = dm.batch_information_fk
                AND c."time" >= dm.dm_start_time
                AND c."time" <= dm.dm_end_time
            GROUP BY c.batch_information_fk, dm.dm_end_time, dm.dm_start_time
        ),
        -- dry mixing ram pressure（step2→step3，与dry_mixing_stats同窗口）
        ram_pressure_dry_mixing AS (
            SELECT c.batch_information_fk,
                AVG(c.ram_pressure) AS avg_ram_pressure_dry_mixing
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN step_times_dry_mixing dm ON c.batch_information_fk = dm.batch_information_fk
                AND c."time" >= dm.dm_start_time
                AND c."time" <= dm.dm_end_time
            GROUP BY c.batch_information_fk
        ),
        -- step7-8 ram pressure（step6→step8，与rotor_speed_7_8同窗口）
        ram_pressure_7_8 AS (
            SELECT c.batch_information_fk,
                AVG(c.ram_pressure) AS avg_ram_pressure_step7_8
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN step_times_7_8 s78 ON c.batch_information_fk = s78.batch_information_fk
                AND c."time" >= s78.step6_time
                AND c."time" <= s78.step8_time
            GROUP BY c.batch_information_fk
        ),
        -- dry mixing end temperature（取 <= dm_end_time 的最近曲线点温度）
        dm_end_max_time AS (
            SELECT c.batch_information_fk, MAX(c."time") AS max_time
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN step_times_dry_mixing dm ON c.batch_information_fk = dm.batch_information_fk
                AND c."time" <= dm.dm_end_time
            GROUP BY c.batch_information_fk
        ),
        dry_mixing_end_temp AS (
            SELECT DISTINCT c.batch_information_fk,  c.temperature AS dry_mixing_end_temp
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN dm_end_max_time dmt ON c.batch_information_fk = dmt.batch_information_fk
                AND c."time" = dmt.max_time
        ),
        -- step10结束前10s：瞬时功率均值、ram pressure均值
        pre_discharge_10s AS (
            SELECT c.batch_information_fk,
                AVG(c."power")      AS avg_power_pre_discharge_10s,
                AVG(c.rotor_speed)  AS avg_rotor_speed_pre_discharge_10s,
                AVG(c.ram_pressure) AS avg_ram_pressure_pre_discharge_10s
            FROM primusmaster."13_production_mixing_bots"."13_02_04_mixing_process_curves_{plant}" c
            JOIN relevant_batches rb ON c.batch_information_fk = rb.batch_information_fk
            JOIN step_times_top st ON c.batch_information_fk = st.batch_information_fk
                AND c."time" >= st.step10_time - 10
                AND c."time" <= st.step10_time
            GROUP BY c.batch_information_fk
        )
        SELECT
            ctd.batch_information_fk,
            ctd.order_id,
            ctd.top_temp_discharge,
            ctd.top_time_discharge,
            ctd.compound_description,
            ctd.compound_name_long,
            ctd.prod_variant,
            ctd.prod_issue,
            ctd.prod_version, 
            bd.bot_temp_discharge,
            bd.bot_time_discharge,
            CAST(pt.power_/1000 AS DECIMAL) AS top_energy_calc_MJ,
            CAST(pt.power_/3600 AS DECIMAL) AS top_energy_kWh,
            CAST(pb.power_/1000 AS DECIMAL) AS bot_energy_calc_MJ,
            CAST(pb.power_/3600 AS DECIMAL) AS bot_energy_kWh,
            s78.step8_time - s78.step6_time AS top_step7_8_duration,
            s78.step8_time AS top_step8_end_time,
            s8et.top_step8_end_temp,
            ct.cooling_start_time AS top_cooling_start_time,
            ct.cooling_end_time AS top_cooling_end_time,
            cts.cooling_start_temp,
            cte.cooling_end_temp,
            rs78.avg_rotor_speed_step7_8,
            rp78.avg_ram_pressure_step7_8,
            dms.dry_mixing_duration,
            CAST(dms.dry_mixing_energy_MJ AS DECIMAL) AS dry_mixing_energy_MJ,
            dms.dry_mixing_avg_power,
            dmet.dry_mixing_end_temp,
            rpdm.avg_ram_pressure_dry_mixing,
            pds.avg_power_pre_discharge_10s,
            pds.avg_rotor_speed_pre_discharge_10s,
            pds.avg_ram_pressure_pre_discharge_10s
        FROM curves_top_discharge ctd
        LEFT JOIN bot_discharge bd ON ctd.batch_information_fk = bd.batch_information_fk
        LEFT JOIN pow_top pt ON ctd.batch_information_fk = pt.batch_information_fk
        LEFT JOIN pow_bot pb ON ctd.batch_information_fk = pb.batch_information_fk
        LEFT JOIN step_times_7_8 s78 ON ctd.batch_information_fk = s78.batch_information_fk
        LEFT JOIN step8_end_temp s8et ON ctd.batch_information_fk = s8et.batch_information_fk
        LEFT JOIN cooling_times ct ON ctd.batch_information_fk = ct.batch_information_fk
        LEFT JOIN cool_temp_start cts ON ctd.batch_information_fk = cts.batch_information_fk
        LEFT JOIN cool_temp_end cte ON ctd.batch_information_fk = cte.batch_information_fk
        LEFT JOIN rotor_speed_7_8 rs78 ON ctd.batch_information_fk = rs78.batch_information_fk
        LEFT JOIN ram_pressure_7_8 rp78 ON ctd.batch_information_fk = rp78.batch_information_fk
        LEFT JOIN dry_mixing_stats dms ON ctd.batch_information_fk = dms.batch_information_fk
        LEFT JOIN dry_mixing_end_temp dmet ON ctd.batch_information_fk = dmet.batch_information_fk
        LEFT JOIN ram_pressure_dry_mixing rpdm ON ctd.batch_information_fk = rpdm.batch_information_fk
        LEFT JOIN pre_discharge_10s pds ON ctd.batch_information_fk = pds.batch_information_fk
    """
    
    cur = conn.cursor()
    cur.execute(sql)
    data = pd.DataFrame(
        cur.fetchall(), columns=['batch_information_fk', 'order_id', 'top_temp_discharge', 'top_time_discharge', 'compound_description', 'compound_name_long', 'prod_variant', 'prod_issue', 'prod_version', 
                                 'bot_temp_discharge', 'bot_time_discharge',
                                 'top_energy_calc_MJ', 'top_energy_kWh',
                                 'bot_energy_calc_MJ', 'bot_energy_kWh',
                                 'top_step7_8_duration', 'top_step8_end_time', 'top_step8_end_temp',
                                 "top_cooling_start_time","top_cooling_end_time","cooling_start_temp","cooling_end_temp",
                                 'avg_rotor_speed_step7_8', 'avg_ram_pressure_step7_8',
                                 'dry_mixing_duration', 'dry_mixing_energy_MJ',
                                 'dry_mixing_avg_power', 'dry_mixing_end_temp',
                                 'avg_ram_pressure_dry_mixing',
                                 'avg_power_pre_discharge_10s', 'avg_rotor_speed_pre_discharge_10s',
                                 'avg_ram_pressure_pre_discharge_10s'])
    cur.close()
    conn.close()

    # 从MMS查询fill factor（top和bottom），按compound_description合并
    sql_ff = f"""
        SELECT
            RecipeID,
            MAX(CASE WHEN ParameterID LIKE '%RHT%' OR ParameterID LIKE '%TOP%' THEN ParameterValue END) AS fill_factor_top,
            MAX(CASE WHEN ParameterID LIKE '%RHB%' OR ParameterID LIKE '%BOT%' THEN ParameterValue END) AS fill_factor_bot
        FROM HESFESFEPLANT.dbo.RecipeCBS3Parameters
        WHERE ParameterName = 'Fill-Factor'
        GROUP BY RecipeID
    """
    conn_mms = connect_mms()
    fill_factors = pd.read_sql(sql_ff, conn_mms)
    conn_mms.close()
    data = data.merge(fill_factors, left_on='compound_description', right_on='RecipeID', how='left')
    data.drop(columns=['RecipeID'], inplace=True, errors='ignore')

    # Replace unreliable Primus bottom discharge temperature with SFE BatchData
    # using the requested OrderID+BatchNumber match.
    data['order_id'] = data['order_id'].astype(str)
    data['BatchNumber'] = pd.to_numeric(
        data['batch_information_fk'].astype(str).str.split('_').str[-1],
        errors='coerce'
    ).astype('Int64')

    sfe_bot_temp = get_sfe_bot_discharge_temperature(
        compound=compound,
        start_date=start_date,
        end_date=end_date,
    )
    if len(sfe_bot_temp) > 0:
        data = data.merge(
            sfe_bot_temp,
            left_on=['order_id', 'BatchNumber'],
            right_on=['OrderID', 'BatchNumber'],
            how='left'
        )
        data['bot_temp_discharge'] = data['bot_temp_discharge_sfe'].combine_first(data['bot_temp_discharge'])
        data.drop(
            columns=['OrderID', 'bot_temp_discharge_sfe', 'bot_temp_timestamp_sfe', 'BatchNumber'],
            inplace=True,
            errors='ignore'
        )

    return data








def MDL_data(
            plant_id='HE',
            all_plants=False,
            db='MUSTANG',
            start_date="2023-01-01",
            end_date="2025-11-01",
            compounds=None,
            #LabExcellence=False,
    ):
    """select the plants you want to include in the list by uncommenting the respective one"""

    plants_list = [
        ["Hefei", "he", 0, "APAC", "HE", M1_QUERY_COMPOUNDS],
    ]
    if db == "PRIMUS":
        steps=["M1"]
        db_data = []

        # just a single plant
        plants_list_ = next((x for x in plants_list if plant_id in x), [])
        Plant = plants_list_[0]
        print(Plant)
        Plant_SHORT_CODE = plants_list_[4]
        #Plant_region=plants_list_[3]
        database=plants_list_[1]

        target_compounds = compounds if compounds is not None else plants_list_[5]
        for comp in target_compounds:
            print("steps: ", steps)
            for step_ in steps:
                df = get_primusdata(
                        plant=database, 
                        compound=comp,
                        step=step_,
                        start_date=start_date,
                        end_date=end_date,
                    )
                # dry mixing columns are now included directly in get_primusdata SQL

                print(Plant +"_"+ step_+ "_" + comp +"_relevant_data_retrieved rows: ", len(df))

                if len(df) > 0:
                    df["Plant"] = Plant
                    df["Compound"] = comp
                    df["plant_short_code"] = Plant_SHORT_CODE
                    df['mixing_step']= step_
                    db_data.append(df)

    elif db == "MUSTANG":

        db_data = []

        # just a single plant
        plants_list_ = next((x for x in plants_list if plant_id in x), [])
        Plant = plants_list_[0]
        print(Plant)
        Plant_SHORT_CODE = plants_list_[4]
        #Plant_region=plants_list_[3]
        database=plants_list_[1]

        target_compounds = compounds if compounds is not None else plants_list_[5]
        for comp in target_compounds:

            df = get_lab_data(
                    plant=database, 
                    compound=comp,
                    start_date=start_date,
                    end_date=end_date,
                )

            print(Plant + "_relevant_data_retrieved rows: ", len(df))
            df["Plant"] = Plant
            df["Compound"] = comp
            df["plant_short_code"] = Plant_SHORT_CODE
            db_data.append(df)


    return pd.concat(db_data, ignore_index=True)


def get_raw_material_properties(plant, polymer=True, start_date='2023-01-01', end_date='2025-01-01'):
    """ #TODO:description"""

    conn = connect_datamart(datamart="mustangmaster")

    sql= f"""
        select *
        from mustangmaster.mustang_global_glue_01.sap_supplier_qm
        where  (plant= """+plant+""")  and material like 'CS100%'
        and "lot creation date" > '2023-01-01'
        and ("short text" like '%heating%' or "short text" like '%fines%')

        """
    if polymer==True:
        sql= f"""
        select *
        from mustangmaster.mustang_global_glue_01.sap_supplier_qm
        where  (plant= """+plant+""")  
        and (material like 'CS100%' or material like 'CE%' or material like 'CN%' or material like 'CR62%' or material like 'CR41%' 
        or material like 'CP%' or material like 'CV%' or material like 'CU%')
        and "lot creation date" > '2023-01-01'
        and (("short text" like '%heating%' and material like 'CS100%') or ("short text" like '%UML 1+4%' and uom ='MOO')) --or "short text" like '%fines%'

        """
    
    cur = conn.cursor()
    cur.execute(sql)
    data = pd.DataFrame(
        cur.fetchall(), columns=['plant',"inspection lot","operation/activity","operation short text","confirmation number","lot creation date","master insp.charac.","short text",
                                 "mean value / s","target value","upper specif. limit",
                                 "lower specif. limit","upplausibility limit","loplausibility limit","material","material description",'uom','vpi','batch','vendor',
                                 "vendor name","unit of measurement",'master','variant',"vendor index"])
    
      
    cur.close()
    conn.close()
    return data


def raw_material_data(
    plant_id='HE',
    all_plants=False,
    start_date="2023-01-01",
    end_date="2025-01-01",    

):
    """
    gets silica raw material properties like moisture and fines content
    """

    plants_list = [
        ["Hefei", "he", "9200", "APAC", "HE", ["M1-T09170", "M1-T33025", "M1-B00458"]],
    ]

    raw_data = []

    # just a single plant
    plants_list_ = next((x for x in plants_list if plant_id in x), [])
    Plant = plants_list_[0]
    print(Plant)
    Plant_SHORT_CODE = plants_list_[4]
    #Plant_region=plants_list_[3]
    #database=plants_list_[1]

    df = get_raw_material_properties(
            plant=plants_list_[2], 
            start_date=start_date,
            end_date=end_date,
        )

    print(Plant + "_relevant_data_retrieved rows: ", len(df))
    df["Plant"] = Plant
    df["plant_short_code"] = Plant_SHORT_CODE
    raw_data.append(df)


    return pd.concat(raw_data, ignore_index=True)   


def polymer_master_MMS_lotIDs(compound):
    """get lostID for solid polymers used in master production to merge with viscosity raw properties"""
    compound_query = _m1_compound_query(compound)

    sql = f"""    

        SELECT DISTINCT
            o.orderID AS FMO_OrderID,
            bm.BatchNumber AS FMO_BatchNumber,
            CASE
                WHEN bm.ChargeID IS NOT NULL AND LEN(bm.ChargeID) >= 4
                    THEN STUFF(bm.ChargeID, 1, 4, '9000')  
                ELSE bm.ChargeID
            END AS polymer_conti_lot_id
        FROM  HESFESFEPLANT.dbo.Orders AS o
        JOIN  HESFESFEPLANT.dbo.BatchMaterials AS bm
            ON o.OrderID = bm.OrderID
        WHERE (o.CompoundName LIKE '{compound_query}%')
        AND ( (bm.MaterialCode LIKE 'CE%' AND bm.MaterialCode NOT LIKE 'CE19%')
            OR bm.MaterialCode LIKE 'CN2%'
            OR bm.MaterialCode LIKE 'CR41%' )

        """
    
    polymer_lotID = pd.read_sql(sql, connect_mms())

    return polymer_lotID

def mixing_traceability_M1_FMF(compound):
    compound_query = _m1_compound_query(compound)

    sql = f"""

        select distinct finals_and_remill_palletID.FMF_OrderID, finals_and_remill_palletID.FMF_BatchWeight,finals_and_remill_palletID.FMF_Recipe, finals_and_remill_palletID.FMF_PalletID, 
        finals_and_remill_palletID.FMF_BatchNumber, finals_and_remill_palletID.FMF_ML, 
        get_M1batches.M1_Recipe,
        get_M1batches.M1_OrderID,get_M1batches.M1_PalletID, get_M1batches.M1_BatchWeight,
        get_M1batches.M1_BatchNumber, 
        get_M1batches.M1_ML, silica_lot.silica_conti_lot, silica_lot.silica_actual_weight, silica_lot.silica_target_weight, silica_lot.silica_tolerance
        from
        (select distinct o.OrderID as FMF_OrderID,  bm.MaterialName, o.CompoundDescription as FMF_Recipe, bh.PalletID as FMF_PalletID, bh.BatchWeight as FMF_BatchWeight, bm.BatchNumber as FMF_BatchNumber, o.Equipment as FMF_ML, bm.ChargeID as M1_PalletID
        from HESFESFEPLANT.dbo.Orders o -- Final orders
        join HESFESFEPLANT.dbo.BatchMaterials bm on o.OrderID= bm.OrderID
        join HESFESFEPLANT.dbo.BatchHeader bh on (bh.OrderID=bm.OrderID and bh.BatchNumber=bm.BatchNumber)
        where (o.CompoundName like '{compound_query}%')
        and (bm.MaterialName like '{compound_query}%')) as finals_and_remill_palletID
        left JOIN 
        (select distinct o.OrderID as M1_OrderID, o.CompoundDescription as M1_Recipe,bh.BatchNumber as M1_BatchNumber, o.Equipment as M1_ML, bh.PalletID as M1_PalletID, bh.BatchWeight as M1_BatchWeight
        from HESFESFEPLANT.dbo.Orders o -- M1 orders
        join HESFESFEPLANT.dbo.BatchHeader bh on o.OrderID = bh.OrderID
        where (o.CompoundName like '{compound_query}%')) as get_M1batches on get_M1batches.M1_PalletID = finals_and_remill_palletID.M1_PalletID
        left JOIN
        (select distinct o.orderID as M1_OrderID, bm.BatchNumber as M1_BatchNumber, o.Equipment as M1_ML, bm.ChargeID as silica_conti_lot, bm.ActualWeight as silica_actual_weight, 
        bm.TargetWeight as silica_target_weight, bm.Tolerance as silica_tolerance
        from HESFESFEPLANT.dbo.Orders o
        join HESFESFEPLANT.dbo.BatchMaterials bm on o.OrderID = bm.OrderID
        where (o.CompoundName like '{compound_query}%')
        and bm.MaterialCode like 'CS100%') as silica_lot on (silica_lot.M1_OrderID = get_M1batches.M1_OrderID and silica_lot.M1_BatchNumber = get_M1batches.M1_BatchNumber)
        order by FMF_OrderID desc, FMF_PalletID, FMF_BatchNumber,  get_M1batches.M1_OrderID, get_M1batches.M1_BatchNumber

        """
    
    M1_FMF = pd.read_sql(sql, connect_mms())

    return M1_FMF


def mixing_traceability_M1_R1_FMF(compound):
    compound_query = _m1_compound_query(compound)
    compound_base = _compound_base(compound)

    sql = f"""

        select distinct finals_and_remill_palletID.FMF_OrderID, finals_and_remill_palletID.FMF_BatchWeight,finals_and_remill_palletID.FMF_Recipe, finals_and_remill_palletID.FMF_PalletID, finals_and_remill_palletID.FMF_BatchNumber, finals_and_remill_palletID.FMF_ML, 
        remill_and_R1_palletID.R1_PalletID,remill_and_R1_palletID.R1_ML,remill_and_R1_palletID.R1_BatchWeight, remill_and_R1_palletID.R1_OrderID,remill_and_R1_palletID.R1_Recipe,
        remill_and_R1_palletID.R1_BatchNumber, get_M1batches.M1_Recipe,
        get_M1batches.M1_OrderID,get_M1batches.M1_PalletID, get_M1batches.M1_BatchWeight,
        get_M1batches.M1_BatchNumber, 
        get_M1batches.M1_ML, silica_lot.silica_conti_lot, silica_lot.silica_actual_weight, silica_lot.silica_target_weight, silica_lot.silica_tolerance
        from
        (select distinct o.OrderID as FMF_OrderID,  o.CompoundDescription as FMF_Recipe, bh.PalletID as FMF_PalletID, bh.BatchWeight as FMF_BatchWeight, bm.BatchNumber as FMF_BatchNumber, o.Equipment as FMF_ML, bm.ChargeID as R1_PalletID
        from HESFESFEPLANT.dbo.Orders o -- Final orders
        join HESFESFEPLANT.dbo.BatchMaterials bm on o.OrderID= bm.OrderID
        join HESFESFEPLANT.dbo.BatchHeader bh on (bh.OrderID=bm.OrderID and bh.BatchNumber=bm.BatchNumber)
        where (o.CompoundName like '{compound_query}%')
        and (bm.MaterialName like 'R1-{compound_base}%')) as finals_and_remill_palletID
        left JOIN 
        (select distinct o.orderID as R1_OrderID, o.CompoundDescription as R1_Recipe,bh.PalletID as R1_PalletID, bh.BatchNumber as R1_BatchNumber, o.Equipment as R1_ML, bm.ChargeID as M1_PalletID, bh.BatchWeight as R1_BatchWeight
        from HESFESFEPLANT.dbo.Orders o -- R1 orders
        join HESFESFEPLANT.dbo.BatchMaterials bm on o.OrderID= bm.OrderID
        join HESFESFEPLANT.dbo.BatchHeader bh on (bh.OrderID=bm.OrderID and bh.BatchNumber=bm.BatchNumber)
        where (o.CompoundName like 'R1-{compound_base}%')
        and (bm.MaterialName like '{compound_query}%')) as remill_and_R1_palletID on remill_and_R1_palletID.R1_PalletID = finals_and_remill_palletID.R1_PalletID
        left JOIN
        (select distinct o.OrderID as M1_OrderID, o.CompoundDescription as M1_Recipe,bh.BatchNumber as M1_BatchNumber, o.Equipment as M1_ML, bh.PalletID as M1_PalletID, bh.BatchWeight as M1_BatchWeight
        from HESFESFEPLANT.dbo.Orders o -- M1 orders
        join HESFESFEPLANT.dbo.BatchHeader bh on o.OrderID = bh.OrderID
        where (o.CompoundName like '{compound_query}%')) as get_M1batches on get_M1batches.M1_PalletID = remill_and_R1_palletID.M1_PalletID
        left JOIN
        (select distinct o.orderID as M1_OrderID, bm.BatchNumber as M1_BatchNumber, o.Equipment as M1_ML, bm.ChargeID as silica_conti_lot, bm.ActualWeight as silica_actual_weight, 
        bm.TargetWeight as silica_target_weight, bm.Tolerance as silica_tolerance
        from HESFESFEPLANT.dbo.Orders o
        join HESFESFEPLANT.dbo.BatchMaterials bm on o.OrderID = bm.OrderID
        where (o.CompoundName like '{compound_query}%')
        and bm.MaterialCode like 'CS100%') as silica_lot on (silica_lot.M1_OrderID = get_M1batches.M1_OrderID and silica_lot.M1_BatchNumber = get_M1batches.M1_BatchNumber)
        order by FMF_OrderID desc, FMF_PalletID, FMF_BatchNumber,  get_M1batches.M1_OrderID, get_M1batches.M1_BatchNumber

        """
    
    M1_R1_FMF = pd.read_sql(sql, connect_mms())

    return M1_R1_FMF


def WO_traceability(compound):
    """ get batch additional info of CM material used"""
    compound_query = _m1_compound_query(compound)
    compound_code_suffix = _compound_base(compound)[1:]

    sql= f"""
        select *
        FROM (
        (select o.OrderID, b.BatchNumber, o.OrderStartTime, o.CompoundDescription, o.CompoundName, r.MaterialName, r.TargetWeight, r.Tolerance, b. ActualWeight, b.MaterialCode, b.ChargeID as input_material_PalletID
        from HESFESFEPLANT.dbo.Orders o
        join HESFESFEPLANT.dbo.BatchMaterials b on o.OrderID = b.OrderID 
        join HESFESFEPLANT.dbo.RecipeMaterials r on (r.RecipeID = o.CompoundDescription and r.MaterialCode = b.MaterialCode) 
        where r.RecipeID like '%5{compound_code_suffix}%' and (b.MaterialCode like 'CM%' or b.MaterialCode like '%WO%')
        and o.CompoundName like '{compound_query}%'
        and o.OrderStatus = 11) as t
        left join
        (select distinct o.orderID as source_of_WO_OrderID, o.CompoundCode as source_of_WO_CompoundCode, o.CompoundName as source_of_WO_CompoundName,o.CompoundDescription as source_of_WO_Recipe,h.PalletID as WO_homogenized_storage_PalletID 
        from HESFESFEPLANT.dbo.Orders o
        left join HESFESFEPLANT.dbo.PalletData h on (o.OrderID = h.OrderID) 
        where (o.CompoundCode like 'CM%' or o.CompoundName like 'FM%')
        ) as t_ on t.input_material_PalletID=t_.WO_homogenized_storage_PalletID
        )

        """
    WO_trac = pd.read_sql(sql, connect_mms())

    return WO_trac  


def master_recipe_phr(compound):
    """ get phr in master recipe for silica, polymer and liquid components"""
    compound_base = _compound_base(compound)

    sql= f"""
        select RecipeID, MaterialCode, MaterialName, Pphr
        FROM  HESFESFEPLANT.dbo.RecipeMaterials r 
        where RecipeID like '%{compound_base}%'
        and (MaterialCode like 'CE%' or MaterialCode like 'CN%' or MaterialCode like 'CR62%' or MaterialCode like 'CR41%' 
        or MaterialCode like 'CP%' or MaterialCode like 'CV%' or MaterialCode like 'CU%'
        or MaterialCode like 'CS%' or MaterialCode like 'CR1009%'
        or MaterialCode like 'CR1109%' or MaterialCode like 'CR1107%' or MaterialCode like 'CR1112%' or MaterialCode like 'CR1117%' 
        or MaterialCode like 'CR1126%' or MaterialCode like 'CR1131%' or MaterialCode like 'CR1134%' or MaterialCode like 'CR1207%'
        or MaterialCode like 'CR1212%' or MaterialCode like 'CR1407%'
        or MaterialName like 'CTP%'
        or MaterialCode like 'CR4113%' or MaterialCode like 'CC0339%' or MaterialCode like 'CA551%')        

        """
    master_phr = pd.read_sql(sql, connect_mms())

    return master_phr


def get_material_actual_weights(compound):
    """get actual material weights per batch from OrderMaterials (BatchWeight = weight of that
    material per one batch) joined with BatchHeader (total batch weight).
    Only keep the material families used later in Mat_Type classification:
    solid elastomer, natural rubber, silica, oil, silane, reclaim, carbon black, and CTP.
    Returns: OrderID, BatchNumber, MaterialCode, MaterialName,
             material_batch_weight, total_batch_weight, weight_ratio_pct, Weight_Type"""

    compound_query = _m1_compound_query(compound)

    sql = f"""
        SELECT
            o.OrderID,
            bh.BatchNumber,
            om.MaterialCode,
            om.MaterialName,
            CAST(om.BatchWeight  AS FLOAT)                                          AS material_batch_weight,
            CAST(bh.BatchWeight  AS FLOAT)                                          AS total_batch_weight,
            CAST(om.BatchWeight  AS FLOAT)
                / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100                    AS weight_ratio_pct
        FROM HESFESFEPLANT.dbo.Orders o
        JOIN HESFESFEPLANT.dbo.OrderMaterials om
            ON o.OrderID = om.OrderID
        JOIN HESFESFEPLANT.dbo.BatchHeader bh
            ON o.OrderID = bh.OrderID
        WHERE o.CompoundName LIKE '{compound_query}%'
          AND (
              (om.MaterialCode LIKE 'CE%' AND om.MaterialCode NOT LIKE 'CE19%')
           OR om.MaterialCode LIKE 'CN%'
           OR om.MaterialCode LIKE 'CR4103%'
           OR om.MaterialCode LIKE 'CS100%'
           OR (om.MaterialCode LIKE 'CS%' AND om.MaterialCode NOT LIKE 'CS100%')
           OR om.MaterialCode LIKE 'CR%'
           OR om.MaterialName LIKE 'CTP%'
            OR om.MaterialCode LIKE 'CC0339%'
            OR om.MaterialCode LIKE 'CA551%'
          )
    """
    df = pd.read_sql(sql, connect_mms())

    material_code = df['MaterialCode'].astype(str)
    material_name = df['MaterialName'].astype(str)

    is_solid_elastomer = material_code.str.contains('CE', na=False) & ~material_code.str.contains('CE19', na=False)
    is_natural_rubber = material_code.str.contains('CN', na=False)
    #is_reclaim = material_code.str.contains('CR4103', na=False)
    is_silica = material_code.str.contains('CS1056|CS1037|CS1008|CS1002', na=False)
    is_oil = material_code.str.contains('CS', na=False) & ~is_silica
    is_silian = material_code.str.contains('CA551', na=False)
    is_carbon_black = material_code.str.contains('CC', na=False)
    #is_liquid_resin = material_code.str.contains('CR', na=False) & ~is_reclaim & ~is_ctp

    df['Weight_Type'] = pd.NA
    df.loc[is_solid_elastomer, 'Weight_Type'] = 'weight_pct_solid_elastomer'
    df.loc[is_natural_rubber, 'Weight_Type'] = 'weight_pct_natural_rubber'
    #df.loc[is_reclaim, 'Weight_Type'] = 'weight_pct_Reclaim'
    df.loc[is_silica, 'Weight_Type'] = 'weight_pct_silica'
    df.loc[is_oil, 'Weight_Type'] = 'weight_pct_oil'
    df.loc[is_silian, 'Weight_Type'] = 'weight_pct_silian'
    #df.loc[is_ctp, 'Weight_Type'] = 'weight_pct_CTP'
    df.loc[is_carbon_black, 'Weight_Type'] = 'weight_pct_carbon_black'

    df = df[df['Weight_Type'].notna()].copy()
    return df


def get_lab_data(plant, compound, start_date='2023-01-01', end_date='2025-11-01'):
    compound_query = _m1_compound_query(compound)

    conn = connect_datamart(datamart="mustangmaster")

    sql = f"""


        SELECT master_recipe, order_id, sample_id, pallet_id, test_code, control_plan_name,
            is_retested, retest_number, test_variable, test_result, test_target,
            test_status_name, test_status, equipment_id, test_order_start_time, prefix,
            tolerance_lower, tolerance_upper, warning_lower, warning_upper,
            disposition_of_compound
    FROM he_datamarts.compound_excellence_datamart
    where compound_name like '{compound_query}%'
    and test_variable = 'MS1+3'
    and test_status <> 'Failed'
    and test_status_name <> 'D'
    and compound_name not like '%M1-X%'
    and equipment_id ='MV5'
    and test_order_start_time between '"""+start_date+"""' and '"""+end_date+"""'

    """
    cur = conn.cursor()
    cur.execute(sql)
    data = pd.DataFrame(
        cur.fetchall(), columns=['master_recipe','order_id', 'sample_id', 'pallet_id', 'test_code','control_plan_name', 'is_retested', 'retest_number', 'test_variable', 'test_result', 'test_target', 
                                'test_status_name', 'test_status', 'equipment_id', 'test_order_start_time', 'prefix', 'tolerance_lower', 'tolerance_upper',
                                'warning_lower',  'warning_upper', 'disposition_of_compound'])
    
    cur.close()
    conn.close()
    print("rows queried: ", len(data))

    return data




def get_sample_batch_120sampling(compound):
    """ get most probable correspondence between lab sampleid and FMF batch nr. when there 120% sampling
    this apparently happens in Lousado, Rayong and KO""" 
    compound_query = _m1_compound_query(compound)

    _sql_= f"""
        select cast(F1.SampleID as int) as SampleID, F1.AtWeight, F1.PalletID, F1.EstimatedWeight, F1.OrderID, F1.BatchNumber, F1.BatchWeight, F2.Nr_of_Total_Batches, F2.total_order_weight
        from
        (
            select SampleID, AtWeight, samples.PalletID, pallets.EstimatedWeight, headers.OrderID, headers.BatchNumber, headers.BatchWeight
            from SFEPLANT.dbo.PalletSample as samples
            join SFEPLANT.dbo.Pallets as pallets on pallets.PalletID=samples.PalletID 
            join SFEPLANT.dbo.Orders as orders on orders.OrderID=pallets.OrderID
            join SFEPLANT.dbo.BatchHeader as headers on (samples.PalletID = headers.PalletID)
            where orders.CompoundName like '{compound_query}%'
            
            union all
            
            select SampleID, AtWeight, samples.PalletID, pallets.EstimatedWeight, headers.OrderID, headers.BatchNumber, headers.BatchWeight
            from SFEPLANT_ARCHIVE.dbo.PalletSample as samples
            join SFEPLANT_ARCHIVE.dbo.Pallets as pallets on pallets.PalletID=samples.PalletID 
            join SFEPLANT_ARCHIVE.dbo.Orders as orders on orders.OrderID=pallets.OrderID
            join SFEPLANT_ARCHIVE.dbo.BatchHeader as headers on (samples.PalletID = headers.PalletID)
            where orders.CompoundName like '{compound_query}%'
        ) as F1
        left join
        (
            select OrderID, max(Nr_of_Total_Batches) as Nr_of_Total_Batches, sum(total_order_weight) as total_order_weight
            from (
                select headers.OrderID, max(headers.BatchNumber) as Nr_of_Total_Batches, sum(headers.BatchWeight) as total_order_weight
                from SFEPLANT.dbo.BatchHeader as headers
                group by headers.OrderID
                union all
                select headers.OrderID, max(headers.BatchNumber) as Nr_of_Total_Batches, sum(headers.BatchWeight) as total_order_weight
                from SFEPLANT_ARCHIVE.dbo.BatchHeader as headers
                group by headers.OrderID
            ) as union_bh
            group by OrderID
        ) as F2 on F1.OrderID=F2.OrderID
    """
    FMF_batches_from_sampleID_120= pd.read_sql(_sql_, connect_mms())
    if len(FMF_batches_from_sampleID_120) == 0:
        return pd.DataFrame(columns=['SampleID', 'AtWeight', 'PalletID', 'EstimatedWeight', 'OrderID', 'BatchNumber', 'BatchWeight', 'Nr_of_Total_Batches', 'total_order_weight', 'batch_number_pred', 'batch_information_fk_final'])
        
    import numpy as np
    
    df = FMF_batches_from_sampleID_120
    sample_ids = df['SampleID']
    pred_base = np.where(sample_ids <= 2, 1, 
                         np.where(sample_ids == 3, 2, 
                                  (sample_ids // 1.2 - 1).astype(int)))
                                  
    max_batch_on_pallet = df.groupby(['OrderID', 'PalletID'])['BatchNumber'].transform('max')
    pred = np.where(sample_ids > 3, np.minimum(pred_base, max_batch_on_pallet), pred_base)
    
    df['batch_number_pred'] = np.minimum(pred, df['Nr_of_Total_Batches']).astype(int)
    df['batch_information_fk_final'] = df['OrderID'].astype(str) + '_' + df['batch_number_pred'].apply(lambda x: str(int(x)).zfill(4)) 

    return df


def get_weighings_out_tolerance(compound):
    """get batches where there was an wheighing out of tolerance registed by code material if the deviaton from target is higher than +/- 1.5%
    and the the amount for the material i the recipe is higher than 5 kg
    """ 

    compound_query = _m1_compound_query(compound)

    sql_ = f"""select distinct t.CompoundName, t.OrderID, t.BatchNumber,t.MaterialCode, t.MaterialName
            from(  
            select orders.CompoundName, materials.OrderID, materials.BatchNumber, materials.MaterialCode, materials.MaterialName,  materials.TargetWeight,materials.Tolerance, (cast(materials.ActualWeight as float)-cast(materials.TargetWeight as float))/cast(materials.TargetWeight as float)*100 as perc_target_dev --, materials.TargetWeight, materials.ActualWeight, materials.Tolerance
                --CompoundDescription,  orders.OrderStartTime, orders.Equipment as MixerLine --recipes.MaterialCode as MaterialCode_rec, 
                from HESFESFEPLANT.dbo.BatchMaterials as materials
                join HESFESFEPLANT.dbo.Orders as orders on materials.OrderID = orders.OrderID 
                where (orders.CompoundName like '{compound_query}%'  and  orders.CompoundName not like '%WO%')
                and  (materials.ActualWeight >  cast(materials.TargetWeight as float) + cast(materials.Tolerance as float)
                or materials.ActualWeight <  cast(materials.TargetWeight as float) - cast(materials.Tolerance as float))                        
           
            ) as t    
            where (t.perc_target_dev > 1.5 or t.perc_target_dev < -1.5)
            and t.TargetWeight > 5
        """
    out_of_tolerance= pd.read_sql(sql_, connect_mms()) 

    return out_of_tolerance

# %%
