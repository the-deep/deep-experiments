import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { useState, useCallback } from "react"

import './style.css';

interface Datum {
  Entry: string;
  'Tagger Name': string;
  Pillars: string;
  Sectors: string;
  'Matched Sentences': string;
  'Predicted Pillars': string;
  'Predicted Sectors': string;
}

interface RowProps {
  item: Datum;
  onSubmit: (value: { feedback: string, entry: string }) => void;
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
      onSubmit({ entry: item['Entry'], feedback });
    },
    [onSubmit, item, feedback],
  );

  return (
    <React.Fragment>
      <tr>
        <td width="30%">{item['Entry']}</td>
        <td>{item['Tagger Name']}</td>
        <td>{item['Pillars']}</td>
        <td>{item['Sectors']}</td>
        <td width="30%">{item['Matched Sentences']}</td>
        <td>{item['Predicted Pillars']}</td>
        <td>{item['Predicted Sectors']}</td>
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

class CustomTableComponent extends StreamlitComponentBase<State> {
  constructor(props: any) {
    super(props);
    const data: Datum[] = JSON.parse(this.props.args["data"]);
    this.state = {
      data,
    };
  }

  handleSubmit = (value: { feedback: string, entry: string }) => {
    Streamlit.setComponentValue(value);
  }

  render() {
    const { data } = this.state;

    return (
      <table>
        <tr>
          <th>Entry</th>
          <th>Tagger Name</th>
          <th>Pillars</th>
          <th>Sectors</th>
          <th>Matched Sentences</th>
          <th>Predicted Pillars</th>
          <th>Predicted Sectors</th>
        </tr>
        {data.map((datum) => (
          <Row
            key={datum.Entry}
            item={datum}
            onSubmit={this.handleSubmit}
          />
        ))}
      </table>
    );
  }
}
export default withStreamlitConnection(CustomTableComponent)
