import {
  withStreamlitConnection,
} from "streamlit-component-lib"
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
      <OfflineTableComponent />
    );
  }
  if (args.componentType === 'OnlineTableComponent') {
    return (
      <OnlineTableComponent />
    );
  }

  return null;
}

export default withStreamlitConnection(EntryPoint);
