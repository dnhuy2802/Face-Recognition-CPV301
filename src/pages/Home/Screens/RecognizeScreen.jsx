import style from "./RecognizeScreen.module.css";
import { Select } from "antd";
import { useEffect, useRef, useMemo } from "react";
import Camera from "../../../components/Camera";
import { getModels, setModel } from "../../../apis/predictService";
import { socket } from "../../../socket";
import { VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS } from "../../../utils/constants";
import Spacer from "../../../components/Spacer";
import { initialState } from "../../../contexts/predictContext";
import useProviderState from "../../../hooks/useProviderState";
import { appStrings } from "../../../utils/appStrings";

function RecognizeScreen() {
  const captureImageCallback = useRef(null);
  const canvasRef = useRef(null);
  const canvasContext = useRef(null);
  const captureInterval = useRef(null);

  // Models
  const store = useMemo(() => useProviderState(initialState), []);
  const models = store((state) => state.models);
  const setModels = store((state) => state.setModels);
  const currentModel = store((state) => state.currentModel);
  const setCurrentModel = store((state) => state.setCurrentModel);

  const setCaptureImageCallback = useMemo(
    () =>
      ({ getScreenshot }) => {
        captureImageCallback.current = getScreenshot;
      },
    []
  );

  function captureImage() {
    const base64Image = captureImageCallback.current();
    socket.emit("recognizing", {
      frame: base64Image,
    });
  }

  function capturingInterval() {
    captureInterval.current = setInterval(() => {
      captureImage();
    }, 1000 / VIDEO_FPS);
  }

  function drawFaces(face) {
    const ctx = canvasContext.current;
    // Reset canvas
    ctx.clearRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
    ctx.beginPath();
    ctx.lineWidth = "2";
    ctx.strokeStyle = "rgb(168, 218, 220)";
    ctx.fillStyle = "rgb(168, 218, 220)";
    ctx.font = "20px Fira Sans";
    face.forEach((face) => {
      const cordinate = face.cordinates;
      ctx.rect(cordinate[0], cordinate[1], cordinate[2], cordinate[3]);
      ctx.fillText(face.name, cordinate[0], cordinate[1]);
    });
    ctx.stroke();
  }

  function handleSetModel(model) {
    setCurrentModel(model);
    setModel(model);
  }

  useEffect(() => {
    // Get canvas context
    canvasContext.current = canvasRef.current.getContext("2d");
    // Open Capture Interval
    capturingInterval();
    // Listen to recognized event
    socket.on("recognized", (res) => {
      if (res.success) {
        drawFaces(res.data);
      } else {
        drawFaces([]);
      }
    });

    // Get models
    getModels().then((data) => {
      setModels(
        data.map((model) => ({ label: model.name, value: model.name }))
      );
      setCurrentModel(data[0].name);
      setModel(data[0].name);
    });

    // Close Capture Interval
    return () => {
      clearInterval(captureInterval.current);
      socket.off("recognized");
    };
  }, []);

  return (
    <div>
      <Select
        className={style.select}
        defaultValue={currentModel}
        options={models}
        placeholder={appStrings.recognize.selectPlaceholder}
        onChange={handleSetModel}
      />
      <Spacer size={10} />
      <div className={style.container}>
        <Camera mediaCallback={setCaptureImageCallback} />
        <div className={style.canvasContainer}>
          <canvas
            id={style.canvas}
            ref={canvasRef}
            width={VIDEO_WIDTH}
            height={VIDEO_HEIGHT}
          ></canvas>
        </div>
      </div>
    </div>
  );
}

export default RecognizeScreen;
