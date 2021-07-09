import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { ReactNode } from "react"
import './style.css';

interface State {
  feedback: string,
  entry: string
}

class CustomTableComponent extends StreamlitComponentBase<State> {

  constructor(props: any) {
    super(props);
    this.state = {entry: '', feedback: ''};
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange = (event: any): void => {
   this.setState({feedback: event.target.value});
  }

  handleSubmit = (entry: any): void => {
    let data: any = {"feedback": this.state.feedback, "entry": entry}
    this.setState(
      () => Streamlit.setComponentValue(data)
    )
  }

  render() {
    var data = JSON.parse(this.props.args["data"])
    const listItems = data.map((item: any, key: number) =>
        <React.Fragment key={"item-" + key}>
          <tr key={key}>
            <td width="30%">{item['Entry']}</td>
            <td>{item['Tagger Name']}</td>
            <td>{item['Pillars']}</td>
            <td>{item['Sectors']}</td>
            <td width="30%">{item['Matched Sentences']}</td>
            <td>{item['Predicted Pillars']}</td>
            <td>{item['Predicted Sectors']}</td>
          </tr>
          <tr key={key}>
            <td colSpan={7}>
                <form onSubmit={()=>this.handleSubmit(item["Entry"])} id={"form-" + key}>
                  <textarea id={"feedback-" + key}  value={this.state.feedback} name="feedback" onChange={this.handleChange}/>
                  <button>Submit Feedback</button>
                </form>
            </td>
          </tr>
        </React.Fragment>
    );
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
            {listItems}
          </table>
    );
  }
}
export default withStreamlitConnection(CustomTableComponent)
