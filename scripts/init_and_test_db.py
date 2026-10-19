"""
Helper script to initialize the PostgreSQL database and create the 'kasparro' DB if necessary.
NEVER print the DATABASE_URL.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from psycopg import sql

load_dotenv()


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not found in .env")
        return

    # Parse target db name or default to 'kasparro'
    # Try connecting to the URL directly first
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            print("Successfully connected directly to configured DATABASE_URL.")
    except psycopg.OperationalError as e:
        err_msg = str(e)
        if "does not exist" in err_msg:
            # Need to create database
            # Try connecting to default 'postgres' database on the same host/port/user to CREATE DATABASE
            print("Target database does not exist. Connecting to 'postgres' DB to create it...")
            # Replace database name at end of URL with 'postgres'
            base_url, _, _ = db_url.rpartition("/")
            admin_url = f"{base_url}/postgres"
            with psycopg.connect(admin_url, autocommit=True) as admin_conn:
                with admin_conn.cursor() as cur:
                    # Extract target DB name
                    target_db = db_url.rpartition("/")[2].split("?")[0]
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db)))
                    print(f"Database '{target_db}' created successfully.")
        else:
            print(f"Connection failed: {err_msg}")
            return

    # Now apply api/schema.sql
    print("Applying api/schema.sql...")
    schema_path = Path("api/schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
    print("Schema applied successfully!")


if __name__ == "__main__":
    main()
