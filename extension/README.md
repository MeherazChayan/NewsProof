# NewsProof Chrome Extension

This extension adds a one-click preview for the current page using the existing NewsProof backend.

## What it does

- Reads the active tab URL
- Sends it to `POST /api/v1/verify/url`
- Shows the verification label and short reason in the popup
- Opens the full result page in the NewsProof web app

## Local development

1. Keep the NewsProof stack running.
2. Open `chrome://extensions`.
3. Turn on `Developer mode`.
4. Click `Load unpacked`.
5. Select the `extension/` folder.

The default local URLs are:

- API: `http://127.0.0.1:8000/api/v1`
- Web app: `http://127.0.0.1:3000`

You can change both from the popup `Settings` button after the extension is loaded.

## Files

- `manifest.json` MV3 extension manifest
- `background.js` active tab lookup, API calls, and short-lived result cache
- `popup.html` extension popup markup
- `popup.css` popup styling
- `popup.js` popup rendering and settings handling
