import style from "./TrainScreen.module.css";
import { useState } from "react";
import { Button, Segmented, Typography } from "antd";
import { FunctionOutlined } from "@ant-design/icons";
import Flex from "../../../components/Flex";
import { trainOptions } from "../../../utils/trainOptions";
import Spacer from "../../../components/Spacer";
import { appStrings } from "../../../utils/appStrings";

function TrainScreen() {
  const [selectedOption, setSelectedOption] = useState(trainOptions[0].value);

  function getTrainOptions() {
    return trainOptions.map((option) => {
      let optionColor = "var(--color-black)";
      if (selectedOption === option.value) {
        optionColor = "var(--color-primary)";
      }
      return {
        label: (
          <div className={style.optionLabel}>
            <option.icon size={45} color={optionColor} />
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

  return (
    <Flex direction="column" align="center">
      <Typography.Title level={3}>{appStrings.train.title}</Typography.Title>
      <Spacer size={20} />
      <Segmented
        className={style.segmented}
        defaultValue={selectedOption}
        options={getTrainOptions()}
        onChange={setSelectedOption}
        size="large"
      />
      <Spacer size={30} />
      <Button
        className={style.trainButton}
        icon={<FunctionOutlined />}
        type="primary"
      >
        {appStrings.train.button}
      </Button>
    </Flex>
  );
}

export default TrainScreen;
