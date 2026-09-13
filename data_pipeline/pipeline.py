import pandas as pd
import os
import sqlite3
from data_pipeline.scraper import scrape_books
from data_pipeline.cleaner import clean_books


# Step 1: Scrape raw data
book_data = scrape_books()

print("Total scraped records:", len(book_data))


# Step 2: Save raw data
raw_df = pd.DataFrame(book_data)

raw_df.to_csv(
    "data_pipeline/data/raw/books_raw.csv",
    index=False
)

print("Raw data saved.")


# Step 3: Clean the data
cleaned_df = clean_books(book_data)


print("Cleaning completed.")


# Step 4: Save cleaned data
cleaned_df.to_csv(
    "data_pipeline/data/processed/books_clean.csv",
    index=False
)

print("Cleaned data saved.")
if os.path.exists("data_pipeline/data/books.db"):
    os.remove("data_pipeline/data/books.db")
conn = sqlite3.connect(
    "data_pipeline/data/books.db"
)
print("SQLite database connected.")
conn.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
)
""")
conn.commit()
print("categories table created")
conn.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
)
""")

conn.commit()

print("Books table created.")
#conn.close()
categories = cleaned_df["category"].unique()

for category in categories:
    conn.execute(
        "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
        (category,)
    )

conn.commit()

print("Categories inserted.")
#print(cleaned_df.columns.tolist())
#print(cleaned_df["in_stock"].head())

for _, row in cleaned_df.iterrows():

    category_id = conn.execute(
        "SELECT category_id FROM categories WHERE category_name = ?",
        (row["category"],)
    ).fetchone()[0]

    conn.execute(
        """
        INSERT INTO books
        (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            int(row["in_stock"]),
            category_id
        )
    )

conn.commit()

print("Books inserted.")
#print(cleaned_df.columns.tolist())
print(pd.read_sql("SELECT COUNT(*) AS total_books FROM books", conn))

print(pd.read_sql("SELECT COUNT(*) AS total_categories FROM categories", conn))

query1 = """
SELECT title, price_gbp, rating
FROM books
WHERE rating = 5
"""

result1 = pd.read_sql(query1, conn)

print("\nQuery 1 - Five star books:")
print(result1)

query2 = """
SELECT title, price_gbp, rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10
"""

result2 = pd.read_sql(query2, conn)

print("\nQuery 2 - 10 most expensive books:")
print(result2)

query3 = """
SELECT DISTINCT rating
FROM books
ORDER BY rating
"""

result3 = pd.read_sql(query3, conn)

print("\nQuery 3 - Distinct ratings:")
print(result3)


query4 = """
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 30
"""

result4 = pd.read_sql(query4, conn)

print("\nQuery 4 - Books priced between £20 and £30:")
print(result4)

query5 = """
SELECT 
    books.title,
    books.price_gbp,
    categories.category_name
FROM books
JOIN categories
ON books.category_id = categories.category_id
LIMIT 10
"""

result5 = pd.read_sql(query5, conn)

print("\nQuery 5 - Books with their categories:")
print(result5)

books_df = pd.read_sql("SELECT * FROM books", conn)

categories_df = pd.read_sql("SELECT * FROM categories", conn)

merge_result = pd.merge(
    books_df,
    categories_df,
    on="category_id"
)

merge_result = merge_result[
    ["title", "price_gbp", "category_name"]
].head(10)

print("\nJOIN result using pd.merge:")
print(merge_result)

sql_result = result5.reset_index(drop=True)
merge_result = merge_result.reset_index(drop=True)

comparison = pd.concat(
    [
        sql_result.add_prefix("SQL_"),
        merge_result.add_prefix("MERGE_")
    ],
    axis=1
)

print("\nSQL JOIN and pd.merge comparison:")
print(comparison)

print("\nDo both results match?")
print(sql_result.equals(merge_result))


with open("data_pipeline/output/query_results.txt", "w", encoding="utf-8") as file:

    file.write("QUERY 1 - SELECT + WHERE\n")
    file.write(query1 + "\n")
    file.write(result1.to_string(index=False))
    file.write("\n\n")

    file.write("QUERY 2 - ORDER BY + LIMIT\n")
    file.write(query2 + "\n")
    file.write(result2.to_string(index=False))
    file.write("\n\n")

    file.write("QUERY 3 - DISTINCT\n")
    file.write(query3 + "\n")
    file.write(result3.to_string(index=False))
    file.write("\n\n")

    file.write("QUERY 4 - BETWEEN\n")
    file.write(query4 + "\n")
    file.write(result4.to_string(index=False))
    file.write("\n\n")

    file.write("QUERY 5 - JOIN\n")
    file.write(query5 + "\n")
    file.write(result5.to_string(index=False))
    file.write("\n\n")

    file.write("SQL JOIN and pd.merge match:\n")
    file.write(str(sql_result.equals(merge_result)))

print("SQL queries and outputs saved.")