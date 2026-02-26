# Claude_2EIS

Fits two EC-Lab EIS datasets and plots data + fits on a shared figure.
The second dataset is offset on the Nyquist plot to avoid overlap.

## Files
- `Claude_2EIS.py`: main script (fits + plots two datasets).
- `run_claude_2eis.py`: tiny runner that calls `main()`.

## Usage
Edit these in `Claude_2EIS.py`:
- `FILEPATH`
- `FILEPATH_2`
- `CYCLE` (or `None` for mid-SOC)
- `NYQUIST_OFFSET`

Then run:

```powershell
python run_claude_2eis.py
```

## Notes
- The parser anchors the header line by searching for `cycle number`.
- Parameter names are normalized so `CPE1-0` / `CPE1_0` variants resolve.

