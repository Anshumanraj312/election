import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objs as go


# Replace the file path with the correct path to your Excel file
file_path = './updated_election_turnout copy.xlsx'

# Read all sheets into a dictionary of DataFrames
excel_file = pd.ExcelFile(file_path)
sheet_names = excel_file.sheet_names[:5]  # Limiting to first 5 sheets
dfs = {sheet_name: excel_file.parse(sheet_name) for sheet_name in sheet_names}
all_sheets = pd.read_excel(excel_file, sheet_name=None)
box_data_columns = [
    'Male_Voter_9AM_percent', 'female_Vote_9AM_percent', 'Total_Vote_9AM_percent',
    'Male_Percent_9AM_to_11AM', 'Female_Percent_9AM_to_11AM', 'Total_Percent_9AM_to_11AM',
    'Male_Percent_11AM_to_1PM', 'Female_Percent_11AM_to_1PM', 'Total_Percent_11AM_to_1PM',
    'Male_Percent_1PM_to_3PM', 'Female_Percent_1PM_to_3PM', 'Total_Percent_1PM_to_3PM',
    'Male_Percent_3PM_to_5PM', 'Female_Percent_3PM_to_5PM', 'Total_Percent_3PM_to_5PM',
    'Total_Percent_5PM_to_final', 'Male_Percent_5PM_to_final', 'Female_Percent_5PM_to_final',
]

# Update the box_data_columns list to include the new columns
box_data_columns += [
    'Male_Vote_9AM', 'female_Vote_9AM', 'Total_Vote_9AM',
    'Male_Vote_9AM_to_11AM', 'Female_Vote_9AM_to_11AM', 'Total_Vote_9AM_to_11AM',
    'Male_Vote_11AM_to_1PM', 'Female_Vote_11AM_to_1PM', 'Total_Vote_11AM_to_1PM',
    'Male_Vote_1PM_to_3PM', 'Female_Vote_1PM_to_3PM', 'Total_Vote_1PM_to_3PM',
    'Male_Vote_3PM_to_5PM', 'Female_Vote_3PM_to_5PM', 'Total_Vote_3PM_to_5PM',
    'Male_Vote_5PM_to_final', 'Female_Vote_5PM_to_final', 'Total_Vote_5PM_to_final',
]

selected_columns = [
    'Male_Vote_9AM_to_11AM', 'Female_Vote_9AM_to_11AM', 'Total_Vote_9AM_to_11AM',
    'Male_Vote_11AM_to_1PM', 'Female_Vote_11AM_to_1PM', 'Total_Vote_11AM_to_1PM',
    'Male_Vote_1PM_to_3PM', 'Female_Vote_1PM_to_3PM', 'Total_Vote_1PM_to_3PM',
    'Male_Vote_3PM_to_5PM', 'Female_Vote_3PM_to_5PM', 'Total_Vote_3PM_to_5PM',
    'Male_Vote_5PM_to_final', 'Female_Vote_5PM_to_final', 'Total_Vote_5PM_to_final',
    'Male_Vote_9AM', 'female_Vote_9AM', 'Total_Vote_9AM'
]
selected_columns2 = selected_columns.copy()

# Add an iframe to embed the Power BI report
power_bi_iframe = html.Div(
    [
        html.Iframe(
            src="https://app.powerbi.com/view?r=eyJrIjoiMGI1NWZlYzQtMjE4OC00YzcxLTlhODUtNGEyZDAxMTY5MmJkIiwidCI6ImVhMWZiMWZiLWYxZTgtNGUzMC04MGE4LTU4NGI5MjI0MGUwMCJ9",
            style={'height': '600px', 'width': '100%'}
        )
    ],
    style={
        'max-width': '1200px', 'margin': '20px auto', 'padding': '20px',
        'border': '1px solid #ddd', 'background-color': '#e0e0e0'
    }
)

# Initialize the Dash app
app = dash.Dash(__name__)

#added by us
server = app.server

# Create dropdown options for sheets
dropdown_options_sheets = [{'label': sheet_name, 'value': sheet_name} for sheet_name in sheet_names]
dropdown_options_sheets2 = dropdown_options_sheets.copy()
dropdown_options_sheets3 = dropdown_options_sheets.copy()
# Create dropdown options for columns (including the new columns)
dropdown_options_columns = [{'label': column, 'value': column} for column in box_data_columns]
# Extract unique PS numbers from the data and limit it to a range of 1-300
ps_numbers = np.arange(1, 301)  # Generating PS numbers from 1 to 300
# Create dropdown options for sheets
dropdown_options_sheets = [{'label': sheet_name, 'value': sheet_name} for sheet_name in sheet_names]

# Add a new dropdown for PS numbers
dropdown_options_psno = [{'label': str(ps), 'value': ps} for ps in ps_numbers]

# Define the article content as a separate variable
article_content = html.Div([
    html.H2("In-Depth Analysis of Voter Turnout in Recent Elections", style={'textAlign': 'center'}),
    html.P("Dataset Overview", style={'fontWeight': 'bold'}),
    html.P(
        "Election is the often called the soul of democracy, and as officials of EC we try to keep this soul as pure as possible through free and fair election's."),
    html.P(
        "The often emphasised word 'FREE and FAIR' demands from us to ever remain so vigilant. The age of DATA give us many tools to do so. This research is an attempt to analyse the post poll data to see how we can bring more fairness and transparency to our process. "),
    html.P("A brief about research DATA", style={'fontWeight': 'bold'}),

    html.P(
        "The analysis report contains 2 primary data sets. First database is the polling station wise demographics data which contains over 50 columns containg details from Age profile, geolocation, critical/vulnurable date, general description, past poll turnouts and much more. This was very helpful in Pre election preperations"),
    html.P(
        "Our second dataset contains a comprehensive analysis of 1,378 polling stations, detailed across 78 columns. It includes crucial information like the number of male, female, and other voters, as well as the total voter count. Additionally, it covers specific data points such as voter turnout percentages and counts at various intervals throughout the election day, ranging from 9 AM to the final count. This was collected through sector officer's on Poll day."),

    html.P("Key Insights", style={'fontWeight': 'bold'}),
    html.Ol([
        html.Li(
            "Gender-Based Voting Patterns: The dataset allows for an analysis of gender-specific voting behaviors. By comparing columns like 'Male_Vote_final' and 'female_Vote_final', we can draw insights into gender participation trends, potentially identifying which gender is more active in the voting process."),
        html.Li(
            "Time-Based Voter Engagement: The data tracks voter turnout at different times of the day, from 9 AM to the final count. This can provide valuable insights into when voters are most likely to turn up at polling stations."),
        html.Li(
            "Analyzing Voter Turnout Percentage: Percentage columns such as 'Male_Percent_final', 'Female_percent_final', and 'total_percent_final' offer a clear view of the proportion of voters who participated out of the total registered voters in each category. This is crucial for understanding voter engagement levels."),
        html.Li(
            "Special Category Voters: The dataset includes information on 'Pwd_Voters' (Persons with Disabilities), offering an important perspective on the inclusivity of the electoral process."),
    ]),
    html.P("Methodology", style={'fontWeight': 'bold'}),
    html.P(
        "The analysis employ statistical methods and data visualization techniques to uncover patterns and trends. For instance, using violin plots to visualize the distribution of voter turnout percentages, or line charts to compare the progression of voter turnout throughout the day. The use of interactive dashboards facilitate dynamic exploration of the data.This method also gives users a chance to play with data and explore some hidden patterns which have thus far remained hidden."),
    html.P("Potential Research Questions", style={'fontWeight': 'bold'}),

    html.P("1. How does voter turnout differ between genders across various polling stations?"),
    html.P(
        "Summary: In the interactive graph 2 we can visualise through the gender wise participation across time and polling station in all five assemblies. "),
    html.P("2. Which time of the day experiences the highest voter turnout?"),
    html.P(
        "Summary: Peak Voting time was usually betwween 9AM - 1PM, male and female peaks differently as can be seen in graph. "),
    html.P(
        "3. Are there noticeable trends in voter turnout post 5PM in some POlling station's. Who are the outliers and what does it signify?"),
    html.P(
        "Summary: This has been the key finding of our research. We found that in many polling stations the voting past 5PM increaed drastically. we have categorised them under outlier's. while there are outlier polling stations across time through the day its significance increases mostly post 5PM. Through our dashboard we can further visualise this rate polling station wise using advanced analytical function."),

    html.P("Conclusion", style={'fontWeight': 'bold'}),
    html.P(
        "This dataset provides a granular view of voter turnout in recent elections, offering insights into gender dynamics, temporal voting patterns, and the inclusivity of the voting process. By delving into this rich dataset, we can gain a deeper understanding of the electoral participation trends, which is essential for strengthening democratic processes. The KEY TAKEAWAY from our research and the suggestion we want to put out is that along with 90/75 criteria, outliers too should be included to CRITICAL PS list and special security measures should be taken as there is high risk that the high last hour turnout is beacause of foul play. I would further like to propose that a standard data format should be developed by commission and such data should be made public to bring transparency and involve people in trend exploration. "),
])
app.layout = html.Div([
    html.Div([
        html.H1("Election Data Analysis", style={'textAlign': 'center'}),

        article_content,

        html.Div([
            dcc.Dropdown(
                id='sheet-dropdown',
                options=dropdown_options_sheets,
                value=sheet_names[0],  # Initial value
                style={'width': '45%', 'margin-right': '5px', 'display': 'inline-block'}
            ),
            dcc.Dropdown(
                id='column-dropdown',
                options=dropdown_options_columns,
                value=box_data_columns[0],  # Initial value
                style={'width': '45%', 'margin-left': '5px', 'display': 'inline-block'}
            ),
        ], style={'width': '100%', 'display': 'flex'}),
        dcc.Graph(id='violin-plot'),
        html.Div(id='outliers-table', style={'margin-top': '20px'}),
    ], style={
        'max-width': '1200px', 'margin': 'auto', 'padding': '20px',
        'border': '1px solid #ddd', 'background-color': '#e0e0e0'
    }),

    html.Div([
        # New heading and content to be added here
        html.H2("Additional Insights with interactive Dashboard", style={'textAlign': 'center'}),
        html.P(
            "You can use this dashboard to analyze the trends better. for example lets say you selected Biaora in violin chart from dropdowns, with time period Total_Vote_5PM_to_final. Amongst the outliers you recieved suppose you want to analyse  बैलास-216 further.Now select it in our dashboard and you can see Voting patterns, previous voting data, demographic records and many key insights. Have fun analysing the data.... "),

        power_bi_iframe,
    ],
        style={'max-width': '1200px', 'margin': '20px auto', 'padding': '20px', 'border': '1px solid #ddd',
               'background-color': '#e0e0e0'}
    ),

    html.Div(
        [
            dcc.Dropdown(
                id='sheet-dropdown2',
                options=dropdown_options_sheets2,
                value=sheet_names[0],  # Initial value
                style={'width': '50%', 'margin-right': '10px'}
            ),

            dcc.Dropdown(
                id='voting-columns-dropdown',
                options=dropdown_options_columns,  # Use the same options as the column dropdown
                value=['Male_Voter_9AM_percent'],  # Initial value as an empty list for multi-select
                multi=True,  # Allow multiple selections
                style={'width': '50%', 'margin-right': '10px'}
            ),

            dcc.Checklist(
                id='toggle-statistics',
                options=[
                    {'label': 'Mean', 'value': 'mean'},
                    {'label': 'Median', 'value': 'median'},
                    {'label': 'Std Dev', 'value': 'std_dev'},
                    {'label': 'Q1', 'value': 'q1'},
                    {'label': 'Q3', 'value': 'q3'},
                    {'label': 'Lower Fence', 'value': 'lower_fence'},
                    {'label': 'Upper Fence', 'value': 'upper_fence'}
                ],
                value=['mean'],  # Initial value (can be an empty list if you want it to start hidden)
                inline=True,
                style={'margin-right': '10px', 'color': '#444'}
            ),

            dcc.Graph(id='aggregate-voting-chart'),
        ],
        style={'max-width': '1200px', 'margin': '20px auto', 'padding': '20px', 'border': '1px solid #ddd',
               'background-color': '#e0e0e0'}
    ),

    # New graph component for displaying sum of values
    html.Div(
        [
            dcc.Dropdown(
                id='sum-values-dropdown',
                options=dropdown_options_sheets3,
                value=sheet_names[1],
                style={'width': '50%', 'margin-right': '10px'}
            ),
            dcc.Dropdown(
                id='selected-columns-dropdown',
                options=[{'label': col, 'value': col} for col in selected_columns],
                value=['Total_Vote_9AM'],  # Initial value as an empty list for multi-select
                multi=True,  # Allow multiple selections
                style={'width': '50%', 'margin-right': '10px'}
            ),
            dcc.Checklist(
                id='toggle-statistics-sum-values',
                options=[
                    {'label': 'Mean2', 'value': 'mean2'},
                    {'label': 'Median2', 'value': 'median2'},
                    {'label': 'Std Dev2', 'value': 'std_dev2'},
                    {'label': 'Q12', 'value': 'q12'},
                    {'label': 'Q32', 'value': 'q32'},
                    {'label': 'Lower Fence2', 'value': 'lower_fence2'},
                    {'label': 'Upper Fence2', 'value': 'upper_fence2'}
                ],
                value=['mean2'],  # Initial value (can be an empty list if you want it to start hidden)
                inline=True,
                style={'margin-right': '10px', 'color': '#333'}
            ),
            dcc.Graph(id='sum-values-line-chart'),
        ],
        style={'max-width': '1200px', 'margin': '20px auto', 'padding': '20px', 'border': '1px solid #ddd',
               'background-color': '#e0e0e0'}
    ),

],
    style={'background-color': '#f0f0f0'})


@app.callback(
    Output('violin-plot', 'figure'),
    [Input('sheet-dropdown', 'value'),
     Input('column-dropdown', 'value')]
)
def update_plot(selected_sheet, selected_column):
    # Generate violin plot for the selected column in the selected sheet
    df = dfs[selected_sheet]
    fig = px.violin(df, y=selected_column, box=True, points="all",
                    title=f'Distribution Plot for {selected_column} - {selected_sheet}',
                    hover_data={'ps_name': True, 'psno': True})
    return fig


# Callback to display outliers in a table
@app.callback(
    Output('outliers-table', 'children'),
    [Input('violin-plot', 'hoverData'),
     Input('sheet-dropdown', 'value'),
     Input('column-dropdown', 'value')]  # Add Input for the column dropdown
)
def display_outliers(hoverData, selected_sheet, selected_column):  # Include selected_column as an input argument

    if hoverData is None or not hoverData['points']:
        return html.Table()  # Return an empty table if hoverData is None or empty

    try:
        # Get the index of the point hovered over in the plot
        point_index = hoverData['points'][0]['pointIndex']
        df = dfs[selected_sheet]

        # Get the selected column name from the violin plot
        curve_number = hoverData['points'][0]['curveNumber']
        column_name = df.columns[curve_number]

        # Get values for the selected column
        column_values = df[selected_column]  # Use selected_column instead of 'Your_Selected_Column'

        # Calculate quartiles and interquartile range (IQR)
        Q1 = np.percentile(column_values, 25)
        Q3 = np.percentile(column_values, 75)
        IQR = Q3 - Q1

        # Calculate lower and upper fences for outliers
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        # Identify lower outliers
        lower_outliers_mask = column_values < lower_fence
        lower_outliers = df[lower_outliers_mask]

        # Retrieve 'ps_name' and 'psno' for the lower outliers
        lower_ps_names_outliers = lower_outliers['ps_name'].tolist()
        lower_psno_outliers = lower_outliers['psno'].tolist()

        # Identify upper outliers
        upper_outliers_mask = column_values > upper_fence
        upper_outliers = df[upper_outliers_mask]

        # Retrieve 'ps_name' and 'psno' for the upper outliers
        upper_ps_names_outliers = upper_outliers['ps_name'].tolist()
        upper_psno_outliers = upper_outliers['psno'].tolist()

        # Format the display of ps_name and psno for lower and upper outliers
        formatted_lower_outliers = [f"{ps_name} ({psno})" for ps_name, psno in
                                    zip(lower_ps_names_outliers, lower_psno_outliers)]
        formatted_upper_outliers = [f"{ps_name} ({psno})" for ps_name, psno in
                                    zip(upper_ps_names_outliers, upper_psno_outliers)]

        return html.Div([
            html.H3(f'Outliers for {selected_column} in {selected_sheet}'),
            html.P(f'Lower Outliers - PS Names: {", ".join(formatted_lower_outliers)}'),
            html.P(f'Upper Outliers - PS Names: {", ".join(formatted_upper_outliers)}')
        ])
    except Exception as e:
        return html.Div(f"Error: {str(e)}")

@app.callback(
    Output('aggregate-voting-chart', 'figure'),
    [Input('sheet-dropdown2', 'value'),
     Input('voting-columns-dropdown', 'value'),
     Input('toggle-statistics', 'value')]  # Include toggle values as input
)


def update_line_plot(selected_sheet, selected_columns, toggle_values):
    # Generate line plots for the selected columns in the selected sheet
    df = dfs[selected_sheet]

    fig = px.line(title=f'Line Plot for AC - {selected_sheet}')

    # Add lines for each selected column
    for column in selected_columns:
        if column in df.columns:
            # Add line trace for each selected column
            fig.add_scatter(x=df.index, y=df[column], mode='lines', name=column)

     # Check toggle values for each statistical measure
    for value in toggle_values:
        if value == 'mean':
            mean_val = np.mean(df[column])
            fig.add_annotation(x=df.index[-1], y=mean_val, text=f'Mean ({mean_val:.2f})',
                                showarrow=False, xshift=5, font=dict(color='black'))
            fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=mean_val, y1=mean_val,
                          line=dict(color='black', width=1, dash='dot'), name=f'{column}_Mean')
        elif value == 'median':
            median_val = np.median(df[column])
            fig.add_annotation(x=df.index[-1], y=median_val, text=f'Median ({median_val:.2f})',
                               showarrow=False, xshift=5, font=dict(color='blue'))
            fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=median_val, y1=median_val,
                          line=dict(color='blue', width=1, dash='dot'), name=f'{column}_Median')
        elif value == 'std_dev':
            std_dev = np.std(df[column])
            fig.add_annotation(x=df.index[-1], y=std_dev, text=f'Std Dev ({std_dev:.2f})',
                               showarrow=False, xshift=5, font=dict(color='green'))
            fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=std_dev, y1=std_dev,
                          line=dict(color='green', width=1, dash='dot'), name=f'{column}_StdDev')
        elif value == 'q1':
            Q1 = np.percentile(df[column], 25)
            fig.add_annotation(x=df.index[-1], y=Q1, text=f'Q1 ({Q1:.2f})',
                               showarrow=False, xshift=5, font=dict(color='orange'))
            fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=Q1, y1=Q1,
                          line=dict(color='orange', width=1, dash='dot'), name=f'{column}_Q1')
        elif value == 'q3':
            Q3 = np.percentile(df[column], 75)
            fig.add_annotation(x=df.index[-1], y=Q3, text=f'Q3 ({Q3:.2f})',
                               showarrow=False, xshift=5, font=dict(color='purple'))
            fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=Q3, y1=Q3,
                          line=dict(color='purple', width=1, dash='dot'), name=f'{column}_Q3')
        elif value == 'lower_fence':
            Q1 = np.percentile(df[column], 25)
            Q3 = np.percentile(df[column], 75)
            IQR = Q3 - Q1
            lower_fence = Q1 - 1.5 * IQR
            fig.add_annotation(x=df.index[-1], y=lower_fence, text=f'Lower Fence ({lower_fence:.2f})',
                               showarrow=False, xshift=5, font=dict(color='red'))
            fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=lower_fence, y1=lower_fence,
                          line=dict(color='red', width=1, dash='dot'), name=f'{column}_LowerFence')
        elif value == 'upper_fence':
            Q1 = np.percentile(df[column], 25)
            Q3 = np.percentile(df[column], 75)
            IQR = Q3 - Q1
            upper_fence = Q3 + 1.5 * IQR
            fig.add_annotation(x=df.index[-1], y=upper_fence, text=f'Upper Fence ({upper_fence:.2f})',
                               showarrow=False, xshift=5, font=dict(color='purple'))
            fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=upper_fence, y1=upper_fence,
                          line=dict(color='purple', width=1, dash='dot'), name=f'{column}_UpperFence')


    # Customizing the layout for hover information
    fig.update_traces(hovertemplate='<b>%{y}</b><br>PS Name: %{customdata[0]}<br>PS No: %{customdata[1]}',
                      customdata=np.column_stack((df['ps_name'], df['psno'])))

    fig.update_layout(xaxis_title='Index', yaxis_title='Votes')

    return fig

@app.callback(
    Output('sum-values-line-chart', 'figure'),
    [
        Input('sum-values-dropdown', 'value'),
        Input('selected-columns-dropdown', 'value'),
        Input('toggle-statistics-sum-values', 'value')
    ]
)
def update_sum_values_chart(selected_sheet, selected_columns, toggle_values):
    # Get the DataFrame for the selected sheet
    df = dfs[selected_sheet]

    # Calculate sum across the selected columns for different time intervals
    sum_values = df[selected_columns].sum()

    # Create a DataFrame for sorting and assigning colors based on values
    sorted_values = sum_values.sort_values(ascending=False)
    sorted_indices = sorted_values.index

    # Assign shades of Sunset colors based on the sorted values
    colors = px.colors.sequential.Sunset  # Choose a Sunset color palette
    color_dict = {col: colors[i % len(colors)] for i, col in enumerate(sorted_indices)}

    # Map the colors to the columns based on their values
    color_series = pd.Series({col: color_dict[col] for col in selected_columns})

    # Create a bar chart for the sum values of the selected columns with Sunset colors
    fig = px.bar(
        x=selected_columns,
        y=sum_values,
        title=f'Vote Distribution in - {selected_sheet}',
        color=color_series,
        color_discrete_map='identity'  # Preserve the color mapping
    )
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Total Votes')

    # Apply light theme (plotly theme)
    fig.update_layout(template='plotly')


    # Calculate aggregate statistical measures based on toggle values for the sum-values-line-chart
    if 'mean2' in toggle_values:
        mean_val = sum_values.mean()
        fig.add_shape(type='line', x0=-0.5, y0=mean_val, x1=len(selected_columns) - 0.5, y1=mean_val,
                      line=dict(color='black', width=2, dash='dash'), name='Mean2')
        fig.add_annotation(x=len(selected_columns) - 0.5, y=mean_val, text=f'Mean2 ({mean_val:.2f})',
                           showarrow=False, xshift=5, font=dict(color='black'))
    if 'median2' in toggle_values:
        median_val = sum_values.median()
        fig.add_shape(type='line', x0=-0.5, y0=median_val, x1=len(selected_columns) - 0.5, y1=median_val,
                      line=dict(color='blue', width=2, dash='dash'), name='Median2')
        fig.add_annotation(x=len(selected_columns) - 0.5, y=median_val, text=f'Median2 ({median_val:.2f})',
                           showarrow=False, xshift=5, font=dict(color='blue'))
    if 'std_dev2' in toggle_values:
        std_dev = sum_values.std()
        fig.add_shape(type='line', x0=-0.5, y0=std_dev, x1=len(selected_columns) - 0.5, y1=std_dev,
                      line=dict(color='green', width=2, dash='dash'), name='Std Dev2')
        fig.add_annotation(x=len(selected_columns) - 0.5, y=std_dev, text=f'Std Dev2 ({std_dev:.2f})',
                           showarrow=False, xshift=5, font=dict(color='green'))
    if 'q12' in toggle_values:
        q1 = np.percentile(sum_values, 25)
        fig.add_shape(type='line', x0=-0.5, y0=q1, x1=len(selected_columns) - 0.5, y1=q1,
                      line=dict(color='orange', width=2, dash='dash'), name='Q12')
        fig.add_annotation(x=len(selected_columns) - 0.5, y=q1, text=f'Q12 ({q1:.2f})',
                           showarrow=False, xshift=5, font=dict(color='orange'))
    if 'q32' in toggle_values:
        q3 = np.percentile(sum_values, 75)
        fig.add_shape(type='line', x0=-0.5, y0=q3, x1=len(selected_columns) - 0.5, y1=q3,
                      line=dict(color='purple', width=2, dash='dash'), name='Q32')
        fig.add_annotation(x=len(selected_columns) - 0.5, y=q3, text=f'Q32 ({q3:.2f})',
                           showarrow=False, xshift=5, font=dict(color='purple'))
    if 'lower_fence2' in toggle_values:
        q1 = np.percentile(sum_values, 25)
        q3 = np.percentile(sum_values, 75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        fig.add_shape(type='line', x0=-0.5, y0=lower_fence, x1=len(selected_columns) - 0.5, y1=lower_fence,
                      line=dict(color='red', width=2, dash='dash'), name='Lower Fence2')
        fig.add_annotation(x=len(selected_columns) - 0.5, y=lower_fence, text=f'Lower Fence2 ({lower_fence:.2f})',
                           showarrow=False, xshift=5, font=dict(color='red'))
    if 'upper_fence2' in toggle_values:
        q1 = np.percentile(sum_values, 25)
        q3 = np.percentile(sum_values, 75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        fig.add_shape(type='line', x0=-0.5, y0=upper_fence, x1=len(selected_columns) - 0.5, y1=upper_fence,
                      line=dict(color='purple', width=2, dash='dash'), name='Upper Fence2')
        fig.add_annotation(x=len(selected_columns) - 0.5, y=upper_fence, text=f'Upper Fence2 ({upper_fence:.2f})',
                           showarrow=False, xshift=5, font=dict(color='purple'))

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
