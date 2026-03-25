# Ongoing Log

## 2026-03-25

- Resolved merge conflicts in `src/infomux/cli.py` and `src/infomux/commands/stream.py`.
- Reviewed inherited audio inventory changes and found two follow-up risks:
  unknown ffmpeg devices were being dropped when `system_profiler` omitted them, and synthetic output-only inventory IDs could be treated as capturable `--output` devices.
- Fixed audio inventory construction to preserve ffmpeg-discovered devices as recordable by default and limited `get_device_by_id()` to real recordable devices only.
- Added regression tests covering unknown ffmpeg inputs and excluding output-only inventory IDs from device lookup.
