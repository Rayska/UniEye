# 📷 UniEye

Handheld camera on the Arduino UNO Q, run entirely on the board with one knob. Click to snap: while Qwen 3.5 0.8B writes a short line about the photo ("Sun — the light of hope"), you adjust an 8x13 LED-matrix thumbnail of it with the knob; then the thumbnail slides aside and you scroll through the line and the temperature at capture time. Long-press to browse saved outputs.

Authors: Ray Sointula, Henrik Pärssinen and Toivo Rannila.

The commands below assume the app was created in App Lab as **UniEye**, so its folder on the
board is `/home/arduino/ArduinoApps/unieye` and its LLM container is
`unieye-llamacpp-models-runner-1`. If your app has another name, adjust them.

## Files

- `app.yaml` - app name, icon and bricks: `arduino:llm` with Qwen 3.5 0.8B, context size 2048
- `python/main.py` - app logic: menu, capture, parallel thumbnail + Qwen, display, history (SQLite)
- `python/matrix_art.py` - photo -> 8x13 LED thumbnail (ported from the handoff's `matrix_sim.py`)
- `python/muse.py` - photo -> text with Qwen + vision (ported from the handoff's `vlm.py`)
- `python/requirements.txt` - `onnxruntime` for the thumbnail's object finder
- `sketch/sketch.ino`, `sketch/sketch.yaml` - MCU side: Modulino Knob/Thermo/Buzzer, LED matrix, Bridge

## Board setup outside the app (needed on a new board)

These live in the board's model folder, not in the app, so they are not in this backup:

1. Install the Qwen 3.5 0.8B model in App Lab (`/var/lib/arduino-app-cli/models/llamacpp/unsloth/Qwen3.5-0.8B-GGUF/`).
2. Vision projector: download `mmproj-F16.gguf` from Hugging Face `unsloth/Qwen3.5-0.8B-GGUF` into that folder,
   then convert it to 8-bit on the board with the LLM runner's own tool (much faster on the UNO Q's cores,
   and saves memory):

   ```
   docker exec unieye-llamacpp-models-runner-1 sh -c 'cd /models/unsloth/Qwen3.5-0.8B-GGUF &&
     LD_LIBRARY_PATH=/opt/pkg-cpu/lib /opt/pkg-cpu/bin/llama-quantize mmproj-F16.gguf mmproj-Q8_0.gguf Q8_0 &&
     mv mmproj-F16.gguf mmproj-F16.gguf.bak'
   ```

   Only one `*mmproj*.gguf` may sit in the folder; the runner then serves `Qwen3.5-0.8B-Q4_0-multimodal`.
3. The object model `u2netp.onnx` (4.6 MB) is downloaded by the app itself into `data/` on first start.

## Runtime data on the board (not in this backup)

`data/history.db` (saved outputs), `data/last_capture.jpg` (latest debug photo), `data/u2netp.onnx`.
Copy them over USB with:

```
~/.arduino15/packages/arduino/tools/adb/32.0.0/adb pull /home/arduino/ArduinoApps/unieye/data/ .
```

## License

The code in this repository is released under the [MIT License](LICENSE).

The models it uses are not included and keep their own licenses: Qwen 3.5 0.8B
(downloaded from Hugging Face `unsloth/Qwen3.5-0.8B-GGUF`) and U2-Net-p (`u2netp.onnx`,
Apache-2.0, from the rembg project).
