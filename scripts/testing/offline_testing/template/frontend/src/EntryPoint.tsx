import React from "react"
import OnlineTableComponent from './OnlineTableComponent';
import OfflineTableComponent from './OfflineTableComponent';

interface Props {
  args: any;
}
function EntryPoint(props: Props) {
  const {
    args
  } = props;

  if (args.componentType === 'OfflineTableComponent') {
    return (
      <OfflineTableComponent
        args={args}
      />
    );
  }
  if (args.componentType === 'OnlineTableComponent') {
    return (
      <OnlineTableComponent
        args={args}
      />
    );
  }

  return null;
}

export default EntryPoint;
