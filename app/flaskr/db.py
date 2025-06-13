import os
import psycopg2
import pickle

def get_db_connection(connection_string):
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(connection_string)
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None
    
def close_db_connection(conn):
    """Closes the database connection."""
    if conn:
        try:
            conn.close()
        except psycopg2.Error as e:
            print(f"Error closing database connection: {e}")
    else:
        print("No connection to close.")

def initialize_db(connection_string):
    """Initializes the database by creating necessary tables."""
    conn = get_db_connection(connection_string)
    if not conn:
        return
    
    try:
        with conn.cursor() as cursor:
            # Example table creation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graffiti_features (
                    image_name VARCHAR(255) NOT NULL, -- e.g., 'graffiti_001.jpg'
                    segment_id INTEGER NOT NULL,      -- Unique ID for each detected graffiti segment in that image
                    feature_vector DOUBLE PRECISION[] NOT NULL,
                    -- You might want to add other metadata here, like:
                    -- bounding_box JSONB,              -- e.g., [x1, y1, x2, y2]
                    mask_data BYTEA NOT NULL,
                    -- detection_confidence REAL,
                    -- created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (image_name, segment_id) -- Ensures unique entry per segment in an image
                );
            """)
            conn.commit()
            print("Database initialized successfully.")
    except psycopg2.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        close_db_connection(conn)

def insert_graffiti_feature(conn, image_name, segment_id, feature_vector, mask, logger):
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO graffiti_features (image_name, segment_id, feature_vector, mask_data)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (image_name, segment_id) DO NOTHING;  -- Prevents duplicate entries
            """, (image_name, segment_id, [float(f) for f in feature_vector], pickle.dumps(mask)))
            conn.commit()
            print(f"Inserted feature for {image_name}, segment {segment_id}.")
    except psycopg2.Error as e:
        logger.info(f"Error inserting graffiti feature: {e}")

def get_all_graffiti_features(conn, logger):
    """Fetches all graffiti features from the database."""
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM graffiti_features;")
            rows = cursor.fetchall()
            return rows
    except psycopg2.Error as e:
        logger.info(f"Error fetching graffiti features: {e}")
        return []
    
def exists_image_name(conn, image_name, logger):
    """Checks if an image name exists in the graffiti_features table."""
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM graffiti_features WHERE image_name = %s LIMIT 1;", (image_name,))
            return cursor.fetchone() is not None
    except psycopg2.Error as e:
        logger.info(f"Error checking image existence: {e}")
        return False
    
def get_by_rowid(conn, labels, target):
    """Fetches graffiti features by row ID."""
    row_indices = [i + 1 for i, label in enumerate(labels) if label == target]
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                WITH numbered AS (
                    SELECT *, ROW_NUMBER() OVER (ORDER BY id) AS row_index
                    FROM my_table
                )
                SELECT *
                FROM numbered
                WHERE row_index = ANY(%s);
                """, (row_indices,))
            return cursor.fetchall()
    except psycopg2.Error as e:
        print(f"Error fetching graffiti feature by row ID: {e}")
        return None