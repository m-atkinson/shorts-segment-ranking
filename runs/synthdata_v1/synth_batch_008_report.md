# Synthetic Batch Report

YAML dir: /Users/matthewatkinson/Apps/shorts-model_v3/transcript-pipeline_full-videos/transcripts
Regressor: runs/train_regressor_minilm_guestnorm_v1/ridge_regressor_guestnorm_v1.pkl
Model: sentence-transformers/all-MiniLM-L6-v2
Selected: 10 / Candidates: 2719
Score threshold (p75): 8.545
Max per transcript: 2
Dedup cosine ≥ 0.95

- 1mulWjODfx1 | guest=Anne Applebaum | score(y_log)=10.313 | src=transcript_anne-applebaum.yaml | chunk=6
- gozRHRZTaqP | guest=Anne Applebaum | score(y_log)=10.305 | src=transcript_anne-applebaum.yaml | chunk=7
- In0K8S0rC3v | guest=Anne Applebaum | score(y_log)=10.301 | src=transcript_anne-applebaum-on-ukraine-europe-and-the-us.yaml | chunk=40
- GtG9vAzeObf | guest=Anne Applebaum | score(y_log)=10.273 | src=transcript_anne-applebaum-on-ukraine-russia-europe-and-the-us.yaml | chunk=36
- uHiIjYQgnso | guest=Anne Applebaum | score(y_log)=10.256 | src=transcript_anne-applebaum-on-ukraine-europe-and-the-us.yaml | chunk=25
- VqMiqBGsC7i | guest=Anne Applebaum | score(y_log)=10.193 | src=transcript_anne-applebaum-on-ukraine-russia-europe-and-the-us.yaml | chunk=4
- 8GZnhJ-CRz- | guest=John Bolton | score(y_log)=9.623 | src=transcript_john-bolton-on-trumps-cabinet-picks-and-what-to-expect-in-his-second-term.yaml | chunk=34
- yXZgTObGnoo | guest=John Bolton | score(y_log)=9.600 | src=transcript_john-bolton-trump-100-days.yaml | chunk=43
- GU1Lngz9JKe | guest=James Carville | score(y_log)=9.545 | src=transcript_james-carville-on-biden-v-trump-2024.yaml | chunk=22
- chUlG-y-c04 | guest=John Bolton | score(y_log)=9.535 | src=transcript_john-bolton-on-trumps-cabinet-picks-and-what-to-expect-in-his-second-term.yaml | chunk=49

## Rejection summary
- below_threshold: 0
- per_transcript_cap: 123
- duplicate: 1
- validate_no_source_file: 0
- validate_no_path: 0
- validate_no_full_text: 0
- validate_guest_mismatch: 0
- validate_empty_chunk: 0
- validate_nav_header: 0
- validate_not_in_source: 0
- too_short: 0