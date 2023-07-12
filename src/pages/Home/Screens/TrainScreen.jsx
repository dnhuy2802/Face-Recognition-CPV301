import style from "./TrainScreen.module.css";
import { useState, useContext, useMemo, useEffect } from "react";
import { Button, Segmented, Typography, InputNumber, Select } from "antd";
import { FunctionOutlined } from "@ant-design/icons";
import Flex from "../../../components/Flex";
import {
  trainOptions,
  mlOptions,
  dlOptions,
} from "../../../utils/trainOptions";
import Spacer from "../../../components/Spacer";
import { appStrings } from "../../../utils/appStrings";
import { StateContext } from "../../../contexts/stateContext";
import { initialState } from "../../../contexts/trainContext";
import useProviderState from "../../../hooks/useProviderState";
import { startTrain } from "../../../apis/trainService";

function TrainScreen() {
  // Selected Option, Machine Learning or Deep Learning
  const [selectedOption, setSelectedOption] = useState(trainOptions[0].value);

  // Provider State
  const { store: globalStore } = useContext(StateContext);
  // Face Options for Training
  const faceOptions = globalStore((state) => state.faceOptions);

  /// Train Store
  const store = useMemo(() => useProviderState(initialState), []);
  const faces = store((state) => state.faces);
  const setFaces = store((state) => state.setFaces);
  // Machine Learning Store
  const mlTrain = store((state) => state.mlTrain);
  const setMlTrain = store((state) => state.setMlTrain);
  const mlTest = store((state) => state.mlTest);
  const setMlTest = store((state) => state.setMlTest);
  const mlAlgorithm = store((state) => state.mlAlgorithm);
  const setMlAlgorithm = store((state) => state.setMlAlgorithm);
  // Deep Learning Store
  const dlTrain = store((state) => state.dlTrain);
  const setDlTrain = store((state) => state.setDlTrain);
  const dlValid = store((state) => state.dlValid);
  const setDlValid = store((state) => state.setDlValid);
  const dlTest = store((state) => state.dlTest);
  const setDlTest = store((state) => state.setDlTest);
  const dlFineTune = store((state) => state.dlFineTune);
  const setDlFineTune = store((state) => state.setDlFineTune);
  const dlBatchSize = store((state) => state.dlBatchSize);
  const setDlBatchSize = store((state) => state.setDlBatchSize);
  const dlNetwork = store((state) => state.dlNetwork);
  const setDlNetwork = store((state) => state.setDlNetwork);

  function handleFaceIdentityChange(values) {
    setFaces(values);
  }

  function handleMlTrainTestChange(train, test) {
    const trainValueChange = train - mlTrain;
    const testValueChange = test - mlTest;
    setMlTrain(train - testValueChange);
    setMlTest(test - trainValueChange);
  }

  function handleDlTrainValidTestChange(train, valid, test) {
    const trainValueChange = train - dlTrain;
    const validValueChange = valid - dlValid;
    const testValueChange = test - dlTest;
    if (trainValueChange !== 0) {
      if (dlValid > dlTest) {
        setDlTrain(train);
        setDlValid(dlValid - trainValueChange);
      } else {
        setDlTrain(train);
        setDlTest(dlTest - trainValueChange);
      }
    }
    if (validValueChange !== 0) {
      if (dlTrain > dlTest) {
        setDlValid(valid);
        setDlTrain(dlTrain - validValueChange);
      } else {
        setDlValid(valid);
        setDlTest(dlTest - validValueChange);
      }
    }
    if (testValueChange !== 0) {
      if (dlTrain > dlValid) {
        setDlTest(test);
        setDlTrain(dlTrain - testValueChange);
      } else {
        setDlTest(test);
        setDlValid(dlValid - testValueChange);
      }
    }
  }

  function onStartTrainingPress() {
    const options =
      selectedOption === trainOptions[0].value
        ? {
            mlTrain,
            mlTest,
            mlAlgorithm,
          }
        : {
            dlTrain,
            dlValid,
            dlTest,
            dlFineTune,
            dlBatchSize,
            dlNetwork,
          };
    startTrain(selectedOption, faces, options);
  }

  /// Set Faces when initializing
  useEffect(() => {
    setFaces(faceOptions.map((option) => option.value));
  }, []);

  function getTrainOptions() {
    return trainOptions.map((option) => {
      let optionColor = "var(--color-black)";
      if (selectedOption === option.value) {
        optionColor = "var(--color-primary)";
      }
      return {
        label: (
          <div className={style.optionLabel}>
            <option.icon size={30} color={optionColor} />
            <Spacer size={10} />
            <Typography.Text strong style={{ color: `${optionColor}` }}>
              {option.name}
            </Typography.Text>
          </div>
        ),
        value: option.value,
      };
    });
  }

  function getFaceIdentitySelect() {
    return (
      <Select
        className={style.select}
        mode="multiple"
        placeholder={appStrings.train.identityPlaceholder}
        // Select all options by default
        defaultValue={faceOptions.map((option) => option.value)}
        options={faceOptions}
        onChange={handleFaceIdentityChange}
      />
    );
  }

  return (
    <Flex direction="column" align="center">
      <Segmented
        className={style.segmented}
        defaultValue={selectedOption}
        options={getTrainOptions()}
        onChange={setSelectedOption}
        size="large"
      />
      <Spacer size={20} />
      {selectedOption === trainOptions[0].value ? (
        /// Machine Learning Options
        <div className={style.optionContainer}>
          <Typography.Title level={5}>
            {appStrings.train.ml.title}
          </Typography.Title>
          <Spacer size={10} />
          <Typography.Text type="secondary">
            {appStrings.train.identityTitle}
          </Typography.Text>
          {getFaceIdentitySelect()}
          <Spacer size={10} />
          <Typography.Text type="secondary">
            {appStrings.train.keywords.train}, {appStrings.train.keywords.test}{" "}
            (%)
          </Typography.Text>
          <Flex>
            <InputNumber
              className={style.inputNumber}
              placeholder={appStrings.train.keywords.train}
              value={mlTrain}
              onChange={(value) => handleMlTrainTestChange(value, mlTest)}
              max={100}
              min={0}
            />
            <Spacer size={10} />
            <InputNumber
              className={style.inputNumber}
              placeholder={appStrings.train.keywords.test}
              value={mlTest}
              onChange={(value) => handleMlTrainTestChange(mlTrain, value)}
              max={100}
              min={0}
            />
          </Flex>
          <Spacer size={10} />
          <Typography.Text type="secondary">
            {appStrings.train.ml.algorithmTitle}
          </Typography.Text>
          <Select
            className={style.select}
            placeholder={appStrings.train.ml.algorithmPlaceholder}
            defaultValue={mlAlgorithm}
            options={mlOptions}
            onChange={setMlAlgorithm}
          />
        </div>
      ) : (
        /// Deep Learning Options
        <div className={style.optionContainer}>
          <Typography.Title level={5}>
            {appStrings.train.dl.title}
          </Typography.Title>
          <Spacer size={10} />
          <Typography.Text type="secondary">
            {appStrings.train.identityTitle}
          </Typography.Text>
          {getFaceIdentitySelect()}
          <Spacer size={10} />
          <Typography.Text type="secondary">
            {appStrings.train.keywords.train}, {appStrings.train.keywords.valid}
            , {appStrings.train.keywords.test} (%)
          </Typography.Text>
          <Flex>
            <InputNumber
              className={style.inputNumber}
              placeholder={appStrings.train.keywords.train}
              value={dlTrain}
              onChange={(value) =>
                handleDlTrainValidTestChange(value, dlValid, dlTest)
              }
              max={100}
              min={0}
            />
            <Spacer size={10} />
            <InputNumber
              className={style.inputNumber}
              placeholder={appStrings.train.keywords.valid}
              value={dlValid}
              onChange={(value) =>
                handleDlTrainValidTestChange(dlTrain, value, dlTest)
              }
              max={100}
              min={0}
            />
            <Spacer size={10} />
            <InputNumber
              className={style.inputNumber}
              placeholder={appStrings.train.keywords.test}
              value={dlTest}
              onChange={(value) =>
                handleDlTrainValidTestChange(dlTrain, dlValid, value)
              }
              max={100}
              min={0}
            />
          </Flex>
          <Spacer size={10} />
          <Typography.Text type="secondary">
            {appStrings.train.dl.fineTuneTitle},{" "}
            {appStrings.train.dl.batchSizeTitle}
          </Typography.Text>
          <Flex>
            <InputNumber
              className={style.inputNumber}
              placeholder={appStrings.train.dl.fineTuneTitle}
              value={dlFineTune}
              onChange={setDlFineTune}
              max={10}
              min={0}
            />
            <Spacer size={10} />
            <InputNumber
              className={style.inputNumber}
              placeholder={appStrings.train.dl.batchSizeTitle}
              value={dlBatchSize}
              onChange={setDlBatchSize}
              max={100}
              min={0}
            />
          </Flex>
          <Spacer size={10} />
          <Typography.Text type="secondary">
            {appStrings.train.dl.networkTitle}
          </Typography.Text>
          <Select
            className={style.select}
            placeholder={appStrings.train.dl.networkPlaceholder}
            defaultValue={dlNetwork}
            options={dlOptions}
            onChange={setDlNetwork}
          />
        </div>
      )}
      <Spacer size={20} />
      <Button
        className={style.trainButton}
        icon={<FunctionOutlined />}
        type="primary"
        size="large"
        onClick={onStartTrainingPress}
      >
        {appStrings.train.button}
      </Button>
    </Flex>
  );
}

export default TrainScreen;
