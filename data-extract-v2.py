import os
import tarfile
import time

folder_path = r"C:\Users\Backwood\Downloads\openwebtext\openwebtext"
output_train = "openwebtext/train_split_v2.txt"
output_val = "openwebtext/val_split_v2.txt"

DOC_BOUNDARY = '\x01'  # reserved doc-boundary char, distinct from UNK ('\x00')

files = sorted(f for f in os.listdir(folder_path) if f.endswith('.xz'))
total_files = len(files)
split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

print(f"total .xz archives: {total_files}, train: {len(files_train)}, val: {len(files_val)}", flush=True)


def process(files_list, out_path):
    start = time.time()
    doc_count = 0
    error_count = 0
    with open(out_path, 'w', encoding='utf-8', newline='\n') as outfile:
        for i, filename in enumerate(files_list):
            file_path = os.path.join(folder_path, filename)
            try:
                with tarfile.open(file_path, mode='r:xz') as tar:
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        raw = f.read()
                        text = raw.decode('utf-8', errors='ignore')
                        outfile.write(text)
                        outfile.write(DOC_BOUNDARY)
                        doc_count += 1
            except Exception as e:
                error_count += 1
                print(f"error processing {filename}: {e}", flush=True)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta_min = (len(files_list) - (i + 1)) / rate / 60 if rate > 0 else float('nan')
                print(f"{out_path}: {i+1}/{len(files_list)} archives | {doc_count} docs | "
                      f"{elapsed/60:.1f} min elapsed | ETA {eta_min:.1f} min", flush=True)

    print(f"{out_path} DONE: {len(files_list)} archives, {doc_count} docs, {error_count} errors, "
          f"{(time.time()-start)/60:.1f} min", flush=True)


process(files_train, output_train)
process(files_val, output_val)
print("EXTRACTION COMPLETE", flush=True)
