-- chflibor.xlsx and chflibor_monthly.xlsx
SELECT * FROM ught_cszh.int_rate;
SELECT ccy,rate_type,MIN(val_date),MAX(val_Date) FROM ught_cszh.int_rate WHERE ccy = 'CHF' GROUP BY ccy, rate_type ORDER BY 3;

SELECT *
  FROM ught_cszh.int_rate
 WHERE ccy = 'CHF'
   AND (rate_type IN ('LIBOR','OBS','3MLIBOR') OR
        (rate_type = '6MLIBOR' and val_date <= '29.jun.2012')
        )
ORDER by val_date,days_fwd;
        
SELECT 'CHFLIBOR' AS RATE_TYPE,val_date,days_fwd,zero_rate
  FROM ught_cszh.int_rate
 WHERE ccy = 'CHF'
   AND (rate_type IN ('LIBOR','OBS','3MLIBOR') OR
        (rate_type = '6MLIBOR' and val_date <= '29.jun.2012')
        )
ORDER by val_date,days_fwd;

SELECT 'CHFLIBOR' AS RATE_TYPE,ir.val_date,ir.days_fwd,ir.zero_rate
  FROM ught_cszh.int_rate ir
 WHERE ir.ccy = 'CHF'
   AND (ir.rate_type IN ('LIBOR','OBS','3MLIBOR') OR
        (ir.rate_type = '6MLIBOR' and ir.val_date <= '29.jun.2012')
        )
   AND ir.val_date = last_day(ir.val_date)
ORDER by ir.val_date,ir.days_fwd;

SELECT 'CHFLIBOR' AS RATE_TYPE,ir.val_date,COUNT(*)
  FROM ught_cszh.int_rate ir
 WHERE ir.ccy = 'CHF'
   AND (ir.rate_type IN ('LIBOR','OBS','3MLIBOR') OR
        (ir.rate_type = '6MLIBOR' and ir.val_date <= '29.jun.2012')
        )
   AND ir.val_date = last_day(ir.val_date)
   AND ir.val_date >= '31.dec.2017'
GROUP BY RATE_TYPE,ir.val_date
ORDER BY ir.val_date;

-- pnl_assets_active.xlsx
SELECT fi.deal_nbr,ac.account_event,ac.ccy,sum(ac.amt),ac.val_date
  FROM ught_cszh.accounting ac
      ,ught_cszh.hedge_rel  hr
      ,ught_cszh.fin_instr  fi
      ,ught_cszh.end_of_day ed
 WHERE ac.account_event IN ('02')
   AND ac.ccy = 'CHF'
   AND ac.hedge_id = hr.hedge_id
   AND fi.fin_instr_id = hr.hedge_item_id
   AND fi.deal_nbr <> 'PARENT'
   AND ed.is_eom = 'Y'
   AND EXTRACT(MONTH FROM ed.val_date) IN (12)
   AND ac.val_date = ed.val_date
GROUP BY ac.val_date,fi.deal_nbr,ac.account_event,ac.ccy
ORDER BY ac.val_date,fi.deal_nbr,ac.account_event
;

-- pnl_assets_historic.xlsx
SELECT fi.deal_nbr,ac.account_event,ac.ccy,sum(ac.amt),ac.val_date
  FROM ught_cszh.accounting ac
      ,ught_cszh.hedge_rel  hr
      ,ught_cszh.fin_instr  fi
      ,ught_cszh.end_of_day ed
 WHERE ac.account_event IN ('11')
   AND ac.ccy = 'CHF'
   AND ac.hedge_id = hr.hedge_id
   AND fi.fin_instr_id = hr.hedge_item_id
   AND fi.deal_nbr <> 'PARENT'
   AND ed.is_eom = 'Y'
   AND EXTRACT(MONTH FROM ed.val_date) IN (12)
   AND ac.val_date = ed.val_date
GROUP BY ac.val_date,fi.deal_nbr,ac.account_event,ac.ccy
ORDER BY ac.val_date,fi.deal_nbr,ac.account_event
;

-- valuations_assets_all.xlsx
SELECT fi.fin_instr_id,pv.payer_receiver_flag,sum(pv.clean_pv),pv.ccy,pv.val_date
  FROM ught_cszh.fin_instr  fi
      ,ught_cszh.end_of_day ed
      ,ught_cszh.actual_pv  pv
 WHERE pv.ccy = 'CHF'
   AND pv.fin_instr_id = fi.fin_instr_id
   AND EXISTS (SELECT * FROM ught_cszh.hedge_rel hr WHERE pv.fin_instr_id = hr.hedge_item_id)
   AND fi.deal_nbr <> 'PARENT'
   AND ed.is_eom = 'Y'
   AND EXTRACT(MONTH FROM ed.val_date) IN (12)
   AND EXTRACT(YEAR FROM ed.val_date) > 2011
   AND pv.val_date = ed.val_date
GROUP BY fi.fin_instr_id,pv.payer_receiver_flag,pv.ccy,pv.val_date
ORDER BY fi.fin_instr_id,pv.val_date,pv.payer_receiver_flag
;

-- valuations_swaps_all.xlsx
SELECT fi.fin_instr_id,pv.payer_receiver_flag,sum(pv.clean_pv),pv.ccy,pv.val_date
  FROM ught_cszh.fin_instr  fi
      ,ught_cszh.end_of_day ed
      ,ught_cszh.actual_pv  pv
 WHERE pv.ccy = 'CHF'
   AND pv.fin_instr_id = fi.fin_instr_id
   AND EXISTS (SELECT * FROM ught_cszh.hedge_rel hr WHERE pv.fin_instr_id = hr.hedge_instr_id)
   AND fi.deal_nbr <> 'PARENT'
   AND ed.is_eom = 'Y'
-- AND EXTRACT(MONTH FROM ed.val_date) IN (12)
-- AND EXTRACT(YEAR FROM ed.val_date) > 2011
   AND pv.val_date = ed.val_date
GROUP BY fi.fin_instr_id,pv.payer_receiver_flag,pv.ccy,pv.val_date
ORDER BY fi.fin_instr_id,pv.val_date,pv.payer_receiver_flag
;

-- valuations_assets_active.xlsx
SELECT fi.fin_instr_id,pv.payer_receiver_flag,sum(pv.clean_pv)-avg(pcf_amt),pv.ccy,pv.val_date
  FROM ught_cszh.fin_instr  fi
      ,ught_cszh.end_of_day ed
      ,ught_cszh.actual_pv  pv
 WHERE pv.ccy = 'CHF'
   AND pv.fin_instr_id = fi.fin_instr_id
   AND EXISTS (SELECT *
                 FROM ught_cszh.hedge_rel hr
                WHERE pv.fin_instr_id = hr.hedge_item_id
                  AND pv.val_date >= hr.hedge_start_date
                  AND (hr.hedge_end_date IS NULL OR
                       hr.hedge_end_date >= pv.val_date
                       )
               )
   AND fi.deal_nbr <> 'PARENT'
   AND ed.is_eom = 'Y'
   AND EXTRACT(MONTH FROM ed.val_date) IN (12)
-- AND EXTRACT(YEAR FROM ed.val_date) > 2011
   AND pv.val_date = ed.val_date
GROUP BY fi.fin_instr_id,pv.payer_receiver_flag,pv.ccy,pv.val_date
ORDER BY fi.fin_instr_id,pv.val_date,pv.payer_receiver_flag
;

-- valuations_swaps_active.xlsx
SELECT fi.fin_instr_id,pv.payer_receiver_flag,sum(pv.clean_pv),pv.ccy,pv.val_date
  FROM ught_cszh.fin_instr  fi
      ,ught_cszh.end_of_day ed
      ,ught_cszh.actual_pv  pv
 WHERE pv.ccy = 'CHF'
   AND pv.fin_instr_id = fi.fin_instr_id
   AND EXISTS (SELECT *
                 FROM ught_cszh.hedge_rel hr
                WHERE pv.fin_instr_id = hr.hedge_instr_id
                  AND pv.val_date >= hr.hedge_start_date
                  AND (hr.hedge_end_date IS NULL OR
                       hr.hedge_end_date >= pv.val_date
                       )
               )
   AND fi.deal_nbr <> 'PARENT'
   AND ed.is_eom = 'Y'
-- AND EXTRACT(MONTH FROM ed.val_date) IN (3,6,9,12)
-- AND EXTRACT(YEAR FROM ed.val_date) > 2011
   AND pv.val_date = ed.val_date
GROUP BY fi.fin_instr_id,pv.payer_receiver_flag,pv.ccy,pv.val_date
ORDER BY fi.fin_instr_id,pv.val_date,pv.payer_receiver_flag
;

-- Analysis Swap Population
SELECT * FROM ught_common.cfg_loading_sel_criteria WHERE entity IN ('CS_ZURICH');
SELECT * FROM ught_cszh.cfg_loading_sel_criteria WHERE entity IN ('CS_ZURICH');

SELECT data_source,CATEGORY,TYPE,COUNT(*)
  FROM ught_common.cfg_loading_sel_criteria
 WHERE entity IN ('CS_ZURICH')
 GROUP BY data_source,CATEGORY,TYPE
 ORDER BY data_source,CATEGORY,TYPE
;

SELECT *
  FROM ught_common.cfg_loading_sel_criteria
 WHERE entity IN ('CS_ZURICH')
   AND TYPE = 'SWAP';

SELECT *
  FROM ught_cszh.cfg_loading_sel_criteria
 WHERE entity IN ('CS_ZURICH')
   AND TYPE = 'SWAP';

SELECT deal_type,COUNT(*) FROM ught_cszh.fin_instr GROUP BY deal_type;

SELECT fi.fin_instr_id,pv.payer_receiver_flag,sum(pv.clean_pv),pv.ccy,pv.val_date
  FROM ught_cszh.fin_instr  fi
      ,ught_cszh.end_of_day ed
      ,ught_cszh.actual_pv  pv
 WHERE pv.ccy = 'CHF'
   AND pv.fin_instr_id = fi.fin_instr_id
   AND NOT EXISTS (SELECT * FROM ught_cszh.hedge_rel hr WHERE pv.fin_instr_id = hr.hedge_instr_id)
   AND fi.deal_nbr <> 'PARENT'
   AND fi.deal_type = 'SWAP'
   AND ed.is_eom = 'Y'
   AND EXTRACT(MONTH FROM ed.val_date) IN (12)
   AND EXTRACT(YEAR FROM ed.val_date) > 2015
   AND pv.val_date = ed.val_date
GROUP BY fi.fin_instr_id,pv.payer_receiver_flag,pv.ccy,pv.val_date
ORDER BY pv.val_date,fi.fin_instr_id,pv.payer_receiver_flag
;

SELECT * FROM ught_cszh.fin_instr_leg_def WHERE fin_instr_id = 5474742;
SELECT * FROM ught_cszh.fin_instr_csfw WHERE fin_instr_id = 5474742 ORDER BY payer_receiver_flag,start_date;

SELECT * FROM ught_cszh.fin_instr_leg_def WHERE fin_instr_id = 3167769;
SELECT * FROM ught_cszh.fin_instr_csfw WHERE fin_instr_id = 3167769 ORDER BY payer_receiver_flag,start_date;
SELECT * FROM ught_cszh.fin_instr_csfw_hist WHERE fin_instr_id = 3167769 ORDER BY payer_receiver_flag,start_date;

SELECT * FROM ught_cszh.hedge_rel WHERE hedge_start_date = '7.may.2020';
SELECT * FROM ught_cszh.fin_instr WHERE fin_instr_id = 5608809;
SELECT * FROM ught_cszh.fin_instr_leg_def WHERE fin_instr_id = 5608809;
SELECT * FROM ught_cszh.fin_instr_csfw WHERE fin_instr_id = 5608809 ORDER BY payer_receiver_flag,start_date;

SELECT * FROM ught_cszh.fin_instr WHERE deal_nbr = '53865685';
SELECT * FROM ught_cszh.hedge_rel WHERE hedge_instr_id = 5335899 AND hedge_end_date IS NULL ORDER BY hedge_start_date DESC;

SELECT * FROM ught_cszh.accounting WHERE hedge_id IN (
SELECT hedge_id FROM ught_cszh.hedge_rel WHERE hedge_instr_id = 5335899 )
  AND account_event IN ('02')
  AND val_date = '30.apr.2020'
 ORDER BY val_date DESC,hedge_id DESC,account_event;

SELECT * FROM ught_cszh.actual_pv WHERE fin_instr_id = 5335899 ORDER BY val_date DESC;
