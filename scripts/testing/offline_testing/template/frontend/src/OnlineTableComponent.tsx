import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { useState, useCallback } from "react"

import './style.css';

interface Datum {
  sentence: string;
  prediction: string;
}

interface RowProps {
  item: Datum;
  onSubmit: (value: { feedback: string, sentence: string, prediction: string }) => void;
}
function Row(props: RowProps) {
  const {
    item,
    onSubmit,
  } = props;

  const [feedback, setFeedback] = useState('');
  const handleFeedbackChange = useCallback(
    (event: any) => {
      setFeedback(event.target.value);
    },
    [],
  );

  const handleSubmit = useCallback(
    () => {
      onSubmit({ sentence: item['sentence'], prediction: item['prediction'], feedback });
    },
    [onSubmit, item, feedback],
  );

  return (
    <React.Fragment>
      <tr>
        <td width="30%">{item['sentence']}</td>
        <td>{item['prediction']}</td>
      </tr>
      <tr>
        <td colSpan={7}>
          <form
            onSubmit={handleSubmit}
          >
            <textarea
              value={feedback}
              name="feedback"
              onChange={handleFeedbackChange}
            />
            <button>
              Submit Feedback
            </button>
          </form>
        </td>
      </tr>
    </React.Fragment>
  );
}

interface State {
  data: Datum[];
}

class OnlineTableComponent extends StreamlitComponentBase<State> {
  constructor(props: any) {
    super(props);
    const data: Datum[] = JSON.parse(this.props.args["data"]);

    console.log("***************************************");
    console.log(this.props.args);
    console.log("***************************************");

    this.state = {
      data,
    };
  }

  handleSubmit = (value: { feedback: string, sentence: string, prediction: string }) => {
    Streamlit.setComponentValue(value);
  }

  render() {
    const { data } = this.state;

    return (
      <table>
        <tr>
          <th>Sentence</th>
          <th>Prediction</th>
        </tr>
        {data.map((datum) => (
          <Row
            key={datum.sentence}
            item={datum}
            onSubmit={this.handleSubmit}
          />
        ))}
      </table>
    );
  }
}
export default withStreamlitConnection(OnlineTableComponent)
