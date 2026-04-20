/*
Does requiring a degree impact salary in remote data science roles?
- Compare salary averages between postings that mention not requiring a degree and those that don't.
*/

SELECT 
    job_no_degree_mention,
    COUNT(*) AS job_count,
    ROUND(AVG(salary_year_avg), 0) AS avg_salary
FROM job_postings_fact
WHERE 
    job_title_short = 'Data Scientist'
    AND salary_year_avg IS NOT NULL
    AND job_work_from_home = True
GROUP BY job_no_degree_mention
ORDER BY avg_salary DESC;
