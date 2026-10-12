# tools

`build_deck.js` regenerates `ISL_Project_Presentation.pptx` from `artifacts/`. Every number
and figure on the slides is read from those JSON/PNG files, so after re-running any
experiment you rebuild the deck rather than editing slides by hand. The "Protocol D —
unseen signer" slide appears automatically once `artifacts/unseen_signer.json` exists.

```bash
npm install pptxgenjs react-icons react react-dom sharp
node tools/build_deck.js ~/Desktop/ISL_Project_Presentation.pptx
```
