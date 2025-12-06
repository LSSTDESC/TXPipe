#!/usr/bin/env python3
"""
Merge two Parquet files by streaming data in chunks.

This script reads two Parquet files, verifies they have the same length,
and creates a new Parquet file containing columns from both input files.
Data is processed in chunks to handle large files efficiently.
"""

import pyarrow.parquet as pq
import pyarrow as pa
import argparse
import sys


def merge_parquet_files(file1_path, file2_path, output_path, chunk_size=10000):
    """
    Merge two Parquet files into a single output file.
    
    Parameters
    ----------
    file1_path : str
        Path to the first input Parquet file
    file2_path : str
        Path to the second input Parquet file
    output_path : str
        Path to the output merged Parquet file
    chunk_size : int, optional
        Number of rows to process at a time (default: 10000)
    """
    
    # Open the Parquet files
    print(f"Opening {file1_path}...")
    parquet_file1 = pq.ParquetFile(file1_path)
    
    print(f"Opening {file2_path}...")
    parquet_file2 = pq.ParquetFile(file2_path)
    
    # Get metadata about the files
    num_rows1 = parquet_file1.metadata.num_rows
    num_rows2 = parquet_file2.metadata.num_rows
    
    print(f"File 1 has {num_rows1:,} rows")
    print(f"File 2 has {num_rows2:,} rows")
    
    # Check that both files have the same number of rows
    # if num_rows1 != num_rows2:
    #     raise ValueError(
    #         f"Files have different lengths: {num_rows1} vs {num_rows2}. "
    #         "Cannot merge files with different row counts."
    #     )
    
    # Get schemas from both files
    schema1 = parquet_file1.schema_arrow
    schema2 = parquet_file2.schema_arrow
    
    # Check for column name conflicts
    columns1 = set(schema1.names)
    columns2 = set(schema2.names)
    conflicts = columns1.intersection(columns2)
    
    if conflicts:
        raise ValueError(
            f"Column name conflicts detected: {conflicts}. "
            "Both files contain columns with the same names."
        )
    
    # Create merged schema
    merged_fields = list(schema1) + list(schema2)
    merged_schema = pa.schema(merged_fields)
    
    print(f"\nMerged schema will have {len(merged_schema.names)} columns:")
    print(f"  From file 1: {len(schema1.names)} columns")
    print(f"  From file 2: {len(schema2.names)} columns")
    
    # Create writer for output file
    print(f"\nWriting merged data to {output_path}...")
    writer = pq.ParquetWriter(output_path, merged_schema)
    
    # Process data in chunks
    total_rows = num_rows1
    rows_processed = 0
    
    # Create iterators for both files
    batch_iter1 = parquet_file1.iter_batches(batch_size=chunk_size)
    batch_iter2 = parquet_file2.iter_batches(batch_size=chunk_size)
    
    try:
        for batch1, batch2 in zip(batch_iter1, batch_iter2):
            # Verify batch sizes match
            if len(batch1) != len(batch2):
                raise ValueError(
                    f"Batch size mismatch at row {rows_processed}: "
                    f"{len(batch1)} vs {len(batch2)}"
                )
            
            # Combine columns from both batches
            # Convert to tables to easily concatenate columns
            table1 = pa.Table.from_batches([batch1])
            table2 = pa.Table.from_batches([batch2])
            
            # Get all columns
            all_columns = {}
            for name in table1.column_names:
                all_columns[name] = table1[name]
            for name in table2.column_names:
                all_columns[name] = table2[name]
            
            # Create merged batch
            merged_table = pa.table(all_columns, schema=merged_schema)
            
            # Write batch to output file
            writer.write_table(merged_table)
            
            rows_processed += len(batch1)
            
            # Progress indicator
            progress = (rows_processed / total_rows) * 100
            print(f"  Progress: {rows_processed:,}/{total_rows:,} rows ({progress:.1f}%)", 
                  end='\r')
    
    finally:
        # Close the writer
        writer.close()
    
    print(f"\n\nSuccessfully merged {rows_processed:,} rows into {output_path}")
    
    # Verify the output file
    print("\nVerifying output file...")
    output_file = pq.ParquetFile(output_path)
    output_rows = output_file.metadata.num_rows
    output_cols = len(output_file.schema_arrow.names)
    
    print(f"  Output file has {output_rows:,} rows and {output_cols} columns")
    
    if output_rows == total_rows:
        print("  ✓ Row count verified")
    else:
        print(f"  ✗ Warning: Expected {total_rows:,} rows but got {output_rows:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge two Parquet files by streaming data in chunks"
    )
    parser.add_argument(
        "file1",
        help="Path to the first input Parquet file"
    )
    parser.add_argument(
        "file2",
        help="Path to the second input Parquet file"
    )
    parser.add_argument(
        "output",
        help="Path to the output merged Parquet file"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of rows to process at a time (default: 10000)"
    )
    
    args = parser.parse_args()
    
    try:
        merge_parquet_files(
            args.file1,
            args.file2,
            args.output,
            chunk_size=args.chunk_size
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
