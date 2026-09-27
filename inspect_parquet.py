import pyarrow.parquet as pq

path = "data/sample_block.bin"

table = pq.read_table(path)

print("Columns:")
print(table.column_names)

print("\nSchema:")
print(table.schema)

print("\nRows:", table.num_rows)

print("\nFirst row:")
print(table.slice(0, 1).to_pydict())