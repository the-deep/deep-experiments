import React from "react"
import CustomTableComponent from './CustomTableComponent';
import CustomTable2Component from './CustomTable2Component';

interface Props {
  args: any;
}
function EntryPoint(props: Props) {
  const {
    args
  } = props;

  if (args.componentType === 'A') {
    return (
      <CustomTableComponent
        args={args}
      />
    );
  }
  if (args.componentType === 'B') {
    return (
      <CustomTable2Component
        args={args}
      />
    );
  }

  return null;
}

export default EntryPoint;
