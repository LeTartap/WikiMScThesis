

SELECT article_id, COUNT(DISTINCT root) as num_roots, COUNT(DISTINCT stratum) as num_strata
FROM all_articles_by_category
GROUP BY article_id
HAVING COUNT(DISTINCT root) > 1 OR COUNT(DISTINCT stratum) > 1;


SELECT articles.title FROM articles WHERE articles.article_id = 35;

SELECT root, stratum FROM all_articles_by_category WHERE article_id = 35;


SELECT article_id, rev_id, COUNT(*)
FROM multiplecategories
GROUP BY article_id, rev_id
HAVING COUNT(*) > 1;


-- Step 1: Create the articles Table

DROP TABLE IF EXISTS dab_dsgnprj_79.articles;
CREATE TABLE articles (
    article_id SERIAL PRIMARY KEY,
    title TEXT UNIQUE
);

-- Step 2: Populate articles with Unique Titles

INSERT INTO articles (title)
SELECT DISTINCT title FROM all_articles_by_category;

-- Step 3: Add article_id to all_articles_by_category

ALTER TABLE all_articles_by_category ADD COLUMN article_id INT;

UPDATE all_articles_by_category a
SET article_id = b.article_id
FROM articles b
WHERE a.title = b.title;
-- Step 3: Add article_id to all_articles_by_category


ALTER TABLE all_articles_by_category
ADD CONSTRAINT pk_article_category PRIMARY KEY (article_id, stratum, root);


ALTER TABLE all_articles_by_category DROP COLUMN title;

-- Step 4: Add article_id to multiplecategories
-- new column to multiplecategories table
ALTER TABLE multiplecategories ADD COLUMN article_id INT;
-- populate article_id in multiplecategories
UPDATE multiplecategories m
SET article_id = a.article_id
FROM articles a
WHERE m.page_title = a.title;

-- Step 5: Add the Foreign Key Constraint
ALTER TABLE multiplecategories
ADD CONSTRAINT fk_article_id
FOREIGN KEY (article_id)
REFERENCES articles(article_id);

-- Step 6: Clean up

ALTER TABLE multiplecategories DROP COLUMN page_title;


ALTER TABLE all_articles_by_category
ADD CONSTRAINT fk_category_article
FOREIGN KEY (article_id)
REFERENCES articles(article_id);
