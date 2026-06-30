select *
        from mustangmaster.mustang_global_glue_01.sap_supplier_qm
        where  (plant= """+plant+""")  
        and (material like 'CS100%' or material like 'CE%' or material like 'CN%' or material like 'CR62%' or material like 'CR41%' 
        or material like 'CP%' or material like 'CV%' or material like 'CU%' or material like 'CC%')
        and "lot creation date" > '2023-01-01'
        and (
            ("short text" like '%heating%' and material like 'CS100%')
            or ("short text" like '%UML 1+4%' and uom ='MOO')
            or (LOWER("short text") like '%nitrogen surface area%' and material like 'CS100%')
            or (LOWER("short text") like '%dbp%' and material like 'CC%')
            or (LOWER("short text") like '%iodine%' and material like 'CC%')
            or (LOWER("short text") like '%stsa%' and material like 'CC%')
            or (LOWER("short text") like '%heating%' and material like 'CC%')
        )
