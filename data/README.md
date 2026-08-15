# Test Splits for Reproduction

This package ships **official test splits only** (no train/dev), matching the paper protocol.

| Dataset | File | Utterances | Labels |
|---------|------|------------|--------|
| IEMOCAP | `IEMOCAP/test.raw.json` | 1,623 | 6 (angry, excited, frustrated, happy, neutral, sad) |
| MELD | `MELD/test.json` | 2,610 | 7 (anger, disgust, fear, joy, neutral, sadness, surprise) |

## Sources and citations

- **IEMOCAP**: Busso et al., *Language Resources and Evaluation*, 2008.  
  Obtain full corpus: [USC SAIL IEMOCAP](https://sail.usc.edu/iemocap/iemocap_download.htm).  
  Use only the official test partition when reproducing this work.

- **MELD**: Poria et al., ACL 2019.  
  Obtain full corpus: [MELD GitHub](https://github.com/declare-lab/MELD).  
  The included `test.json` follows the official MELD test split.

## License note

Redistribution of these test files is for **research reproducibility** only. If you publish or share derivatives, cite the original dataset papers and comply with each dataset's license terms.
