SELECT TOP (1000) [OrderID]
      ,[BatchNumber]
      ,[EquipmentID]
      ,[SequenceNumber]
      ,[MaterialCode]
      ,[MaterialName]
      ,[ChargeID]
      ,[TargetWeight]
      ,[ActualWeight]
  
      ,[Timestamp]
  FROM [SFEPLANT_ARCHIVE].[dbo].[BatchMaterials] WITH (NOLOCK)
  WHERE (
          (MaterialCode LIKE 'CE%' AND MaterialCode NOT LIKE 'CE19%')
       OR MaterialCode LIKE 'CN%'
       OR MaterialCode LIKE 'CC%'
       OR MaterialCode LIKE 'CS100%')

       and ChargeID not like 'Unknown Lot'