import style from "./UploadScreen.module.css";
import { Space, Button, Typography } from "antd";
import { AiFillCamera } from "react-icons/ai";
import { appStrings } from "../../../utils/appStrings";
import Spacer from "../../../components/Spacer";
import Grid from "../../../components/Grid";
import FaceCard from "../../../components/FaceCard";

function UploadScreen() {
  return (
    <Space direction="vertical" block className={style.container}>
      <Button type="dashed" className={style.captureButton}>
        <Space direction="vertical" size={1}>
          <AiFillCamera size={24} />
          {appStrings.upload.captureButton}
        </Space>
      </Button>
      <Spacer size={10} />
      <Typography.Title level={5}>
        {appStrings.upload.registeredTitle}
      </Typography.Title>
      <Grid>
        <FaceCard
          id="toyeucauratnhieu"
          imgUrl="https://scontent.fsgn2-7.fna.fbcdn.net/v/t1.15752-9/357920184_1636379756844579_1003244497613936086_n.jpg?_nc_cat=108&ccb=1-7&_nc_sid=ae9488&_nc_ohc=JWVu7LDW3vgAX86_2B7&_nc_ht=scontent.fsgn2-7.fna&oh=03_AdS2bLHwDSBRNgWm0V4DywRLWgQtRMDtgDtZR8mzSym29g&oe=64C92885"
          name="Đinh Trần Yến Vy"
        />
      </Grid>
    </Space>
  );
}

export default UploadScreen;
