# Synthetic Batch Report

YAML dir: data/raw/transcripts
Regressor: runs/train_regressor_minilm_guestfeat_v1/ridge_regressor_guestfeat_v1.pkl
Model: sentence-transformers/all-MiniLM-L6-v2
Selected: 10 / Candidates: 2719
Score threshold (p75): 8.753
Max per transcript: 2
Dedup cosine ≥ 0.95

- ODjfcRNL2ED | guest=Anne Applebaum | score(y_log)=10.291 | src=transcript_anne-applebaum-on-ukraine-europe-and-the-us.yaml | chunk=40
- LbdDZ1c5jAU | guest=Anne Applebaum | score(y_log)=10.238 | src=transcript_anne-applebaum.yaml | chunk=7
- 2rjTbrNLwMt | guest=Anne Applebaum | score(y_log)=10.229 | src=transcript_anne-applebaum-on-ukraine-russia-europe-and-the-us.yaml | chunk=36
- shF6PwKluYI | guest=Anne Applebaum | score(y_log)=10.189 | src=transcript_anne-applebaum-on-ukraine-russia-europe-and-the-us.yaml | chunk=5
- FdlKdMwj6uU | guest=Anne Applebaum | score(y_log)=10.184 | src=transcript_anne-applebaum-on-ukraine-europe-and-the-us.yaml | chunk=25
- vtaiJVfU7wi | guest=Anne Applebaum | score(y_log)=10.151 | src=transcript_anne-applebaum.yaml | chunk=6
- cpHdEoziIbo | guest=John Bolton | score(y_log)=9.666 | src=transcript_john-bolton-on-trumps-cabinet-picks-and-what-to-expect-in-his-second-term.yaml | chunk=50
- b_y6ShRfh2z | guest=James Carville | score(y_log)=9.517 | src=transcript_james-carville-on-biden-v-trump-2024.yaml | chunk=22
- ucR_LGOTU2I | guest=John Bolton | score(y_log)=9.514 | src=transcript_john-bolton-trump-100-days.yaml | chunk=27
- xw7gBOirOl3 | guest=John Bolton | score(y_log)=9.514 | src=transcript_john-bolton-on-trumps-cabinet-picks-and-what-to-expect-in-his-second-term.yaml | chunk=34

## Rejection summary
- below_threshold: 0
- per_transcript_cap: 126
- duplicate: 0
- validate_no_source_file: 0
- validate_no_path: 0
- validate_no_full_text: 0
- validate_guest_mismatch: 0
- validate_empty_chunk: 0
- validate_nav_header: 0
- validate_not_in_source: 0
- too_short: 0