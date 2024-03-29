import os
import random
from datetime import datetime
from time import sleep

from dotenv import load_dotenv
from postgresql_client import PostgresSQLClient

load_dotenv()

TABLE_NAME = "data"
NUM_ROWS = 20


def main():
    pc = PostgresSQLClient(
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

    # Get all columns from the data table
    try:
        columns = pc.get_columns(table_name=TABLE_NAME)
        print(columns)
    except Exception as e:
        print(f"Failed to get schema for table with error: {e}")

    # Load smoke data
    with open("sentences.txt", "r") as f:
        sentences = f.readlines()

    # Loop over all columns and create random values
    for i in range(NUM_ROWS):
        # Randomize values for feature columns
        # Add speech_id and current time
        speech_id = random.randint(1, 4)
        data = [
            speech_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sentences[i].strip()]
        # Insert data
        query = f"""
            insert into {TABLE_NAME} ({",".join(columns)})
            values {tuple(data)}
        """
        pc.execute_query(query)
        sleep(2)


if __name__ == "__main__":
    main()
