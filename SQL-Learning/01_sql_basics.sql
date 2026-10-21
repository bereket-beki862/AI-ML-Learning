
-- Create a database if it does not already exist
IF DB_ID('Beki_test2') IS NULL
    CREATE DATABASE Beki_test2;
GO

USE Beki_test2;
GO

-- Create a Students table
IF OBJECT_ID('dbo.Students', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.Students (
        ID INT PRIMARY KEY,
        Name VARCHAR(50),
        Age INT,
        Marks DECIMAL(5,2)
    );
END;
GO

-- Insert sample records
IF NOT EXISTS (SELECT 1 FROM dbo.Students WHERE ID = 1)
    INSERT INTO dbo.Students VALUES (1, 'Alice', 20, 85);

IF NOT EXISTS (SELECT 1 FROM dbo.Students WHERE ID = 2)
    INSERT INTO dbo.Students VALUES (2, 'Bob', 21, 78);

IF NOT EXISTS (SELECT 1 FROM dbo.Students WHERE ID = 3)
    INSERT INTO dbo.Students VALUES (3, 'Charlie', 19, 92);
GO

-- Display all students
SELECT * FROM dbo.Students;
GO
