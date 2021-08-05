--'C:/Projects/Masters_studies/Thesis/Clinical-GAN1/data/PRESCRIPTIONS.csv'



COPY (
select subject_id,hadm_id,gsn,ndc,drug from mimic_hosp.prescriptions where  ndc<>'0' and ndc is not null and ndc <>'' and length(ndc)=11 order by subject_id,starttime)
TO 'C:/Projects/Masters_studies/Thesis/Clinical-GAN1/data/PRESCRIPTIONS.csv' WITH (FORMAT CSV, HEADER);



COPY (
with t as(
select *,
 ROW_NUMBER() OVER (
      PARTITION BY B.hadm_id
      ORDER BY B.seq_num
   ) row_num
from 
mimic_hosp.diagnoses_icd B 
)

select * from t)
 TO 'C:/Projects/Masters_studies/Thesis/Clinical-GAN1/data/DIAGNOSES_ICD.csv' WITH (FORMAT CSV, HEADER);


COPY (
with t as(
select *,
 ROW_NUMBER() OVER (
      PARTITION BY B.hadm_id
      ORDER BY B.seq_num
   ) row_num
from 
mimic_hosp.procedures_icd B 
)

select * from t)
 TO 'C:/Projects/Masters_studies/Thesis/Clinical-GAN1/data/PROCEDURES_ICD.csv' WITH (FORMAT CSV, HEADER);


-- Fetching the records of patient associated with their admissions information such as age,insurance, gender etc. Ignoring infants data.

COPY (
	
with tmp as (

SELECT 	
	ad.*
	,EXTRACT(DAYS FROM ad.admittime - (to_timestamp(concat(pa.anchor_year::text,'-01-01 00:00:00'),'YYYY-MM-DD HH24:MI:SS'))) + pa.anchor_age * 365 as age

FROM mimic_core.admissions ad
LEFT OUTER JOIN mimic_core.patients pa
ON ad.subject_id = pa.subject_id order by subject_id,admittime 
) 	
	
select * from tmp where age > 2 * 365
)
TO 'C:/Projects/Masters_studies/Thesis/Clinical-GAN1/data/ADMISSIONS.csv' WITH (FORMAT CSV, HEADER);


