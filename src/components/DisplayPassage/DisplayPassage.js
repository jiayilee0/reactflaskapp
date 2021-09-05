import React from "react";
import { Empty, Typography } from "antd";
import "./styles.css";

export const DisplayPassage = ({ passage }) => {
  const formatPassage = (passage) => {
    const formattedPassage = passage.split("\n");
    return formattedPassage.map((paragraph) => (
      <Typography.Paragraph>{paragraph}</Typography.Paragraph>
    ));
  };

  return passage ? (
    <Typography.Paragraph>{formatPassage(passage)}</Typography.Paragraph>
  ) : (
    <Empty
      image={Empty.PRESENTED_IMAGE_SIMPLE}
      description="No entered passage"
    />
  );
};
