SELECT  order_id, pallet_id, sample_id, equipment_id, test_variable, test_code, test_variable_extended, test_status, test_status_name, test_result_start_time, test_result_end_time, test_result_start_time_utc, test_result_end_time_utc, retest_number, test_name, test_result, test_target, uom, compound_name, alias, alias_suffix, prod_variant, prod_issue, prod_version, prod_group, compound_type, compound_switch, test_order_start_time, test_order_start_time_utc, plant_id, tolerance_lower, tolerance_upper, warning_lower, warning_upper, corp_tolerance_lower, corp_tolerance_upper, corp_warning_lower, corp_warning_upper, control_plan_category_name, control_plan_name, is_retested, subsample_name, disposition_of_compound, quality_state_id, mixer, prefix, master_recipe, suffix, kpi_relevant, additional_test_kpi_relevant, additional_test_none_kpi_relevant, is_consumed, produced_qty, remaining_qty, change_id, weighting_factor_fk, test_order_shift_date, test_result_shift_date, uid, precalc_fk, created_timestamp, last_modified_timestamp, datamart_pk, hundred_days_start, extend_reduce_hundred_days, hundred_days_flag, toi_dil_id, tri_dil_id, tv_dil_id, tr_dil_id, tr_tv_dil_id, tollow_dil_id, tr_tollow_dil_id, organization_fk
FROM he_datamarts.compound_excellence_datamart
where test_variable = 'MS1+3'
and test_result_start_time >= '2024-01-01'
and test_status <> 'Failed'
and test_status_name <> 'D'
and compound_name not like '%M1-X%'
and equipment_id = 'MV5'


