# Description: This file contains the functions that will be used in throughout the 3 notebooks created for the code part of the master thesis.
import Imports as imps


# Function to "pivot" certain columns of a dataframe:
def pivot_columns(df: imps.pd.DataFrame,
                  columns_to_pivot: imps.List[str],
                  id_column: str,
                  max_columns: int = 3
) -> imps.pd.DataFrame:
    """
    This functions transforms a dataset by pivoting certain columns, that is decomposing them according to their different values for
    a same id_column.

    Parameters:
    - df: Dataframe to transform.
    - columns_to_pivot: List of columns that should be pivoted.
    - id_column: Column that will be the unique identifier of the transformed dataframe.
    - max_columns: Maximum number of pivoted columns that will be created for each column in columns_to_pivot.
    """
    df = df.copy()

    df = df.groupby(id_column).apply(
        lambda group: imps.pd.Series({
            **{f"{col}_{i+1}": group.iloc[i][col] if i < len(group) else None 
            for col in columns_to_pivot for i in range(max_columns)},
            **{col: group.iloc[0][col] for col in df.columns if col not in columns_to_pivot and col != id_column}
        })
    ).reset_index()

    return df


# Function to compute a bar plot, with the possibility of enhancing a specific value through a different colour
def generate_barplot(df: imps.pd.DataFrame,
                     column: str,
                     title: str,
                     xaxis_label: str,
                     yaxis_label: str = "Count",
                     enhanced_value: str = None
):
    """
    This functions creates a bar plot from the input data, and also allows to enhance a certain value for the column.

    Parameters:
    - df: Dataframe containing the data to visualize.
    - column: Column to plot.
    - enhanced_value: Optional value that should be enhanced in the bar plot.
    - xaxis_label: Label for the x-axis of the bar plot.
    - yaxis_label: Label for the y-axis of the bar plot.
    - title: Title of the bar plot.
    """    
    df = df.copy()

    colors = ["#BFD62F" if x == enhanced_value else "#5C666C" for x in df[column].value_counts().index]
    
    fig = imps.px.bar(x = df[column].value_counts().index,
                      y = df[column].value_counts().values,
                      color = df[column].value_counts().index,
                      color_discrete_sequence = colors,
                      labels = {"x": xaxis_label, "y": yaxis_label},
                      title = title,
                      height = 600)

    fig.update_layout(title_x = 0.5, title_font = dict(color = "black"), xaxis_title_font = dict(color = "black"),
                      yaxis_title_font = dict(color = "black"), showlegend = False, template = "plotly_white")
    fig.show()


# Function to compute a bar plot for application status with respect to other categorical variable
def status_by_barplot(df: imps.pd.DataFrame,
                      status_by_column: str,
                      status_by_label: str,
                      enhanced_value: str = None
):
    """
    This functions creates a bar plot for the admission status, detailed with respect to another categorical variable.

    Parameters:
    - df: Dataframe to transform.
    - status_by_column: Column of interest to detail the admission status.
    - status_by_label: Label for the column of interest.
    - enhanced_value: Optional value that should be enhanced in the bar plot.
    """    
    df = df.copy()

    status_by = df.groupby([status_by_column, "DEstadoPT"]).size().reset_index(name = "count")

    fig = imps.px.bar(
        status_by,
        x = status_by_column,
        y = "count",
        color = "DEstadoPT",
        barmode = "group",
        labels = {"DEstadoPT": "Application Status", "count": "Count", status_by_column: status_by_label},
        title = f"Applicants' Application Status by {status_by_label}",
        color_discrete_sequence = ["#BFD62F" if x == enhanced_value else "#5C666C" for x in status_by["DEstadoPT"].unique()],
        height = 600
    )

    fig.update_layout(title_x = 0.5, title_font = dict(color = "black"), xaxis_title_font = dict(color = "black"),
                    yaxis_title_font = dict(color = "black"), showlegend = False, template = "plotly_white")
    fig.show()


# Function to return the time of the day depending on the hour
def time_of_day(hour: int) -> str:
    """
    This functions returns the time of the day in 4 different categories, depending on the associated hour.

    Parameters:
    - hour: Hour of the day.
    """     
    if 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 20:
        return "Afternoon"
    elif 20 <= hour < 24:
        return "Evening"
    

# Function to check if there are rows that are not missing with prior levels missing
def missing_levels(df: imps.pd.DataFrame,
                   variable: str,
                   n_levels: int
) -> imps.pd.DataFrame:
    """
    This functions checks if there are values missing for a certain level, but the value is not missing for the level that comes afterwards.
    It returns the rows that do not respect the desired condition.

    Parameters:
    - df: Dataframe where the columns belong.
    - variable: Variable of interest to check if all levels are compliant.
    - n_levels: Maximum number of levels that the variable has.
    """
    conditions = []
    for i in range(1, n_levels):
        lower_level = f"{variable}_{i}"
        higher_levels = [f"{variable}_{j}" for j in range(i + 1, n_levels + 1)]
        condition = df[lower_level].isna() & df[higher_levels].notna().any(axis = 1)
        conditions.append(condition)

    overall_condition = imps.pd.concat(conditions, axis = 1).any(axis = 1)

    return df[overall_condition]


# Function to retrieve the maximum degree of a certain applicant
def max_academic_degree(row: imps.pd.Series,
                        ordered_list: list) -> str:
    """
    This functions returns the highest academic degree attained by a certain applicant.

    Parameters:
    - row: Series representing a particular obervation in a dataframe.
    - ordered_list: List of academic degrees in descending order.
    """
    for element in ordered_list:
        if element in row.values:
            return element
    return None


# Function to create a 100% stacked by chart
def stacked_bar_chart_100(df: imps.pd.DataFrame,
                          analyse: str,
                          analyse_by: str,
                          analyse_label: str,
                          analyse_by_label: str,
                          title: str):
    """
    This functions plots a 100% stacked bar chart, exploring the relationship between 2 categorical variables.

    Parameters:
    - df: Dataframe where the columns belong.
    - analyse: Variable to be on the x-axis of the graph.
    - analyse_by: Variable to analyse data by (the one that will be displayed in different colors in the graph).
    - analyse_label: Label to describe the "analyse" variable in the graph.
    - analyse_by_label: Label to describe the "analyse_by" variable in the graph.
    - title: Title of the graph.
    """    
    df = df.copy()

    data = df.groupby([analyse_by, analyse]).size().reset_index(name = "Count")
    data["Percentage"] = data.groupby(analyse_by)["Count"].transform(lambda x: x / x.sum() * 100)

    colors = ["#BFD62F", "#A7C957", "#8CB369", "#56A05C", "#4C9A2A", "#2E6E20", "#1D4E1F", "#0F3D19"]
    categories = data[analyse].unique()
    color_discrete_map = {category: colors[i % len(colors)] for i, category in enumerate(categories)}

    fig = imps.px.bar(
        data,
        x = analyse_by,
        y = "Percentage",
        color = analyse,
        title = title,
        labels = {analyse_by: analyse_by_label, analyse: analyse_label},
        barmode = "stack",
        height = 600,
        color_discrete_map = color_discrete_map
    )

    fig.update_layout(title_x = 0.5, title_font = dict(color = "black"), xaxis_title_font = dict(color = "black"),
                    showlegend = False, template = "plotly_white", yaxis_title = None, yaxis_showticklabels = False)
    fig.show()


# Function to print all values of a certain column
def print_all_values(df: imps.pd.DataFrame,
                     column: str,
                     values_per_line: int = 5,
                     separator: str = " || "):
    """
    This functions prints all unique values of a certain column, also referring their number and how many missing values there are.

    Parameters:
    - df: Dataframe where the columns belong.
    - column: Column of interest to analyse the values.
    - values_per_line: Number of values to show in each printed row.
    - separator: String to separate each of the printed values.
    """ 
    df = df.copy()

    index = df[column].value_counts().index.tolist()

    for i in range(0, len(index), values_per_line):
        print(separator.join(map(str, index[i:i + values_per_line])))

    print(f"\nUnique values for '{column}':", len(df[column].unique()))
    print(f"\nMissing values for '{column}':", df[column].isna().sum())


# Function to create a conditional column based on the presence of a desired value(s) in other columns
def create_conditional_column (df: imps.pd.DataFrame,
                               columns_to_find: list,
                               new_column: str,
                               values_to_find: list,
                               positive_value,
                               negative_value
) -> imps.pd.DataFrame:
    """
    This functions creates a conditional column based on the presence of a desired target value(s) in a group of columns of the dataset

    Parameters:
    - df: Dataframe where the columns belong and where the new column will be added.
    - columns_to_find: List of columns on which to find the desired value(s).
    - new_column: Name of the conditional column to add to the dataset.
    - values_to_find: List of desired value(s) to find in the list of columns.
    - positive_value: Value to assign to the new column if the condition returns positive.
    - negative_value: Value to assign to the new column if the condition returns negative.
    """      
    df = df.copy()
    
    df[new_column] = df.apply(
        lambda row: positive_value if any(
            any(value.lower() in str(row[column]).lower() for value in values_to_find) for column in columns_to_find
        ) else negative_value,
        axis = 1)
    
    return df


# Function to check how many values can be converted to float
def is_float(value: str):
    """
    Check if a value can be converted to a float, considering that "." or "," can be decimal separators.
    """
    try:
        float(str(value).replace(",", "."))
        return True
    except ValueError:
        return False


# Function to find from how many rows could bee retrieved 2 float numbers, representing maximum and minimum grades (created with the help of ChatGPT)
def count_rows_with_two_floats(series: imps.pd.Series):
    """
    Count the number of rows in a pandas Series where exactly two floats can be extracted from the text.

    Parameters:
    - series: A pandas Series containing textual data where numbers (floats) may be present.
    """
    def extract_two_floats(text: str):
        text = str(text).replace(",", ".")  # Normalize decimal separators
        numbers = imps.re.findall(r'\d+(?:\.\d+)?', text)  # Find all numbers with decimals
        return len(numbers) == 2  # Check if there are exactly two numbers

    return series.apply(extract_two_floats).sum()

# Function to disconsider student as a professional activity
def remove_student_values(df: imps.pd.DataFrame) ->  imps.pd.DataFrame:
    """
    Removes all data about professional activities that involve being a student, replacing it with missing values.
    This will apply for the rows about the job description ("ActivProFuncao"), employer (ActivProEntidadePatronal),
    start (DataInicio) and end (DataFim) dates.

    Parameters:
    - df: Pandas dataframe where the columns to transform are located.
    """
    df = df.copy()

    active_funcao_columns = ["ActivProFuncao_1", "ActivProFuncao_2", "ActivProFuncao_3",
                             "ActivProFuncao_4", "ActivProFuncao_5", "ActivProFuncao_6"]
    other_columns = {
        "DataInicio": ["DataInicio_1", "DataInicio_2", "DataInicio_3", "DataInicio_4", "DataInicio_5", "DataInicio_6"],
        "DataFim": ["DataFim_1", "DataFim_2", "DataFim_3", "DataFim_4", "DataFim_5", "DataFim_6"],
        "ActivProEntidadePatronal": ["ActivProEntidadePatronal_1", "ActivProEntidadePatronal_2", "ActivProEntidadePatronal_3",
                                     "ActivProEntidadePatronal_4", "ActivProEntidadePatronal_5", "ActivProEntidadePatronal_6"]}
    
    for idx, funcao_col in enumerate(active_funcao_columns):
        mask = df[funcao_col].str.contains(r"(?i)\b(Estudante|Student)\b", na = False)
        df.loc[mask, funcao_col] = None

        for key, cols in other_columns.items():
            df.loc[mask, cols[idx]] = None

    return df


# Function to identify if the applicant was employed at the time of the application
def working_upon_application(row: imps.pd.Series,
                             start_date_columns: list,
                             end_date_columns: list,
                             positive_value,
                             negative_value):
    """
    Describes, in a binary format, if a certain applicant was working at the moment of their application.

    Parameters:
    - row: Dataframe row to analyse.
    - start_date_columns: List of the names of the variables related to the starting date of the professional experience.
    - end_date_columns: List of the names of the variables related to the ending date of the professional experience.
    - positive_value: Value to assign if the applicant was working at the moment of their application.
    - negative_value: Value to assign if the applicant was not working at the moment of their application.
    """    
    for start_col, end_col in zip(start_date_columns, end_date_columns):
        start_date = row[start_col]
        end_date = row[end_col]
        if imps.pd.notna(start_date) and imps.pd.isna(end_date):
            return positive_value
    
    max_end_date = max([row[end_col] for end_col in end_date_columns if imps.pd.notna(row[end_col])], default = None)
    if max_end_date and max_end_date > row["datacandidaturafim"]:
        return positive_value
    return negative_value


# Function to compute a histogram, with the possibility of enhancing a specific value through a different colour
def generate_histogram(df: imps.pd.DataFrame,
                       column: str,
                       title: str,
                       xaxis_label: str,
                       yaxis_label: str = "Count",
                       bin_size: int = None,
                       enhanced_value: str = None):
    """
    This function creates a histogram from the input data, and also allows to enhance a certain value for the column.

    Parameters:
    - df: Dataframe containing the data to visualize.
    - column: Column to plot.
    - title: Title of the histogram.
    - xaxis_label: Label for the x-axis of the histogram.
    - yaxis_label: Label for the y-axis of the histogram.
    - bin_size: Size of the bins for the histogram.
    - enhanced_value: Optional value that should be enhanced in the histogram.
    """
    df = df.copy()

    unique_values = df[column].dropna()
    colors = ["#BFD62F" if x == enhanced_value else "#5C666C" for x in unique_values]

    fig = imps.px.histogram(df, x = column, nbins = bin_size, color_discrete_sequence = colors,
                            labels = {"x": xaxis_label, "y": yaxis_label}, title = title, height = 600)
    fig.update_traces(marker_line_width = 1, marker_line_color = "white")
    
    fig.update_layout(title_x = 0.5, title_font = dict(color = "black"), xaxis_title_font = dict(color = "black"),
                      yaxis_title_font = dict(color = "black"), showlegend = False, template = "plotly_white")
    fig.show()


# Helper function to calculate the final GPA for masters' students (NOT IN USE)
def calculate_gpa_for_masters(df_group: imps.pd.DataFrame,
                              courses_of_interest: list):
    """
    Helps to calculate the final GPA for masters' students.

    Parameters:
    - df: Grouped dataframe for a certain combination of "id_individuo" and "nm_curso_pt".
    - courses_of_interest: List of courses of interest.
    """
    df_group = df_group.copy()

    excluded = df_group[df_group["ds_discip_pt"].isin(courses_of_interest)]
    non_excluded = df_group[~df_group["ds_discip_pt"].isin(courses_of_interest)]

    avg_non_excluded = non_excluded["notaFinalDisciplina"].mean()

    if not excluded.empty:
        final_avg = round((avg_non_excluded + excluded["notaFinalDisciplina"].mean()) / 2, 2)
    else:
        final_avg = round(avg_non_excluded, 2)
    return final_avg


# Function to calculate the final GPA for all students (NOT IN USE)
def calculate_final_gpa(df: imps.pd.DataFrame,
                        courses_of_interest: list):
    """
    Calculates the final GPA of a certain student taking a certain program.

    Parameters:
    - df: Grouped dataframe for a certain combination of "id_individuo" and "nm_curso_pt".
    - courses_of_interest: List of courses of interest.
    """
    df = df.copy()
    final_gpa_list = []

    for (id_individuo, nm_curso_pt), group in df.groupby(["id_individuo", "nm_curso_pt"]):
        if nm_curso_pt.startswith("P"):  # Posgraduate programs
            final_gpa = round(group["notaFinalDisciplina"].mean(), 2)
        elif nm_curso_pt.startswith("M"):  # Masters' programs
            final_gpa = calculate_gpa_for_masters(group, courses_of_interest)
        else:
            final_gpa = None

        group["FinalGPA"] = final_gpa
        final_gpa_list.append(group)
    
    return imps.pd.concat(final_gpa_list)


# Function to compute summary statistics to relate a numerical and a categorical variable
def summary_statistics(df: imps.pd.DataFrame,
                       categorical_column: str,
                       numerical_column: str):
    """
    Computes the average, median and standard deviation of a numerical variable, both independently and for all values of a certain categorical variable.

    Parameters:
    - df: Grouped dataframe for a certain combination of "id_individuo" and "nm_curso_pt".
    - courses_of_interest: List of courses of interest.
    """   
    df = df.copy()

    print(f"Average {numerical_column}:", round(df[numerical_column].mean(), 2))
    print(f"Median {numerical_column}:", round(df[numerical_column].median(), 2))
    print(f"{numerical_column} Standard Deviation:", round(df[numerical_column].std(), 2))

    print("\n--------------------------------\nAverage:\n", df.groupby(categorical_column)[numerical_column].mean())
    print("\n--------------------------------\nMedian:\n", df.groupby(categorical_column)[numerical_column].median())
    print("\n--------------------------------\nStandard Deviation:\n", df.groupby(categorical_column)[numerical_column].std())


# Function to compute a boxplot
def generate_boxplot(df: imps.pd.DataFrame,
                     numerical_column: str,
                     categorical_column: str,
                     title: str,
                     xaxis_label: str,
                     yaxis_label: str):
    """
    Creates a boxplot from the input data.

    Parameters:
    - df: Dataframe containing the data to visualize.
    - column: Column to plot.
    - title: Title of the histogram.
    - xaxis_label: Label for the x-axis of the histogram.
    - yaxis_label: Label for the y-axis of the histogram.
    """
    df = df.copy()

    fig = imps.px.box(df, x = categorical_column, y = numerical_column, title = title,
                      labels = {categorical_column: xaxis_label, numerical_column: yaxis_label},
                      color_discrete_sequence = ["#5C666C"], height = 600)
    
    fig.update_layout(title_x = 0.5, title_font = dict(color = "black"), xaxis_title_font = dict(color = "black"),
                      yaxis_title_font = dict(color = "black"), showlegend = False, template = "plotly_white")
    fig.show()


# Function to compute a scatter plot
def generate_scatterplot(df: imps.pd.DataFrame,
                         x_column: str,
                         y_column: str,
                         title: str,
                         xaxis_label: str,
                         yaxis_label: str,
                         color_by_column: str = None):
    """
    Creates a scatter plot from the input data.

    Parameters:
    - df: DataFrame containing the data.
    - x_column: Column name to place in the x-axis.
    - y_column: Column name to place in the y-axis.
    - title: Title of the scatter plot.
    - xaxis_label: Label for the x-axis of the scatter plot.
    - yaxis_label: Label for the y-axis of the scatter plot.
    """
    df = df.copy()

    fig = imps.px.scatter(df, x = x_column, y = y_column, title = title, labels = {x_column: xaxis_label, y_column: yaxis_label},
                          color = color_by_column, color_discrete_sequence = ["#5C666C", "#BFD62F"], height = 600)
    
    fig.update_layout(title_x = 0.5, title_font = dict(color = "black"), xaxis_title_font = dict(color = "black"),
                      yaxis_title_font = dict(color = "black"), template = "plotly_white")
    fig.show()


# Function to do a fuzzy left join between df_applicants and HEI rankings (created with the help of ChatGPT)
def fuzzy_merge(df_left: imps.pd.DataFrame,
                df_right: imps.pd.DataFrame,
                left_key: str,
                right_key: str,
                suffix: str,
                similarity_percentage: int = 80)  ->  imps.pd.DataFrame:
    """
    Performs a fuzzy left join of two datasets, accepting to merge columns that have over a certain percentage of similarity.

    Parameters:
    - df_left: Left dataset for the merge.
    - df_right: Right dataset for the merge.
    - left_key: Column of the left dataset used on the merge.
    - right_key:  Column of the right dataset used on the merge.
    - suffix: Suffix to add to the columns of df_right after the merge.
    """
    df_left = df_left.copy()
    df_right = df_right.copy()

    # Auxiliary function to define the rules for the merging of the datasets
    def match_institution(name):
        if imps.pd.isna(name):
            return None
        
        name = str(name)
        match = imps.process.extractOne(name, df_right[right_key].astype(str), scorer = imps.fuzz.ratio)

        if match and match[1] > similarity_percentage:
            return match[0]
        return None

    df_left[f"{left_key}_matched"] = df_left[left_key].apply(match_institution)
    
    df = df_left.merge(df_right, how = "left", left_on = f"{left_key}_matched", right_on = right_key)
    
    df.rename(lambda x: x + suffix if x in df_right.columns else x, axis = 1, inplace = True)
    df.drop(columns = [f"{left_key}_matched"], inplace = True)
    
    return df


# Function to convert string column to numerical
"""
def convert_to_numeric(df: imps.pd.DataFrame,
                       column: str) -> imps.pd.DataFrame:
    ""
    Converts a column of a dataframe from string to numeric, handling commas and final marks as decimal separators, as well as and
    non-numeric characters.

    Parameters:
    - df: Dataframe where the column belongs.
    - column: Column to convert to numeric.
    ""
    df = df.copy()

    df[column] = (df[column].astype(str).str.replace(",", ".", regex = False).replace(r"[^\d.]", "", regex = True))
    df[column] = imps.pd.to_numeric(df[column], errors = "coerce")
    return df
"""
def convert_to_numeric(df: imps.pd.DataFrame,
                       column: str) -> imps.pd.DataFrame:
    """
    Extracts the first float from a string column, handling both '.' and ',' as decimal separators.

    Parameters:
    - df: Dataframe where the column belongs.
    - column: Column to extract the float from.
    """
    df = df.copy()

    # Auxiliary function to extract the first float from a string
    def extract_first_float(value):
        if imps.pd.isna(value):
            return imps.np.nan
        
        # Replace commas with dots and search for the first float
        value = str(value).replace(",", ".")
        match = imps.re.search(r"\d+\.\d+|\d+", value)  # Finds first float or int
        
        return float(match.group()) if match else imps.np.nan

    df[column] = df[column].apply(extract_first_float)
    return df



# Function to extract the first two floats from a string column, and create two new columns with these values
def extract_scale(df: imps.pd.DataFrame,
                   column_to_transform: str,
                   min_col_name: str,
                   max_col_name: str) -> imps.pd.DataFrame:
    """
    Creates two new columns in a dataframe, containing the first and last floats extracted from a string column.

    Parameters:
    - df: Dataframe where the columns belong.
    - column_to_transform: Column to extract the floats from.
    - min_col_name: Name of the new column to store the minimum float.
    - max_col_name: Name of the new column to store the maximum float.
    """
    df = df.copy()

    # Auxiliary function to extract the first and last floats from a string value
    def extract_floats(value):
        if imps.pd.isna(value):
            return [None, None]
        
        value = str(value)
        value = value.replace(",", ".")
        floats = imps.re.findall(r"\d+\.\d+|\d+", value)
        floats = [float(x) for x in floats]
        
        if len(floats) >= 2:
            return [min(floats), max(floats)]
        elif len(floats) == 1:
            return [None, floats[0]]
        else:
            return [None, None]

    df[[min_col_name, max_col_name]] = imps.pd.DataFrame(df[column_to_transform].apply(extract_floats).tolist(), index = df.index)

    swap_condition = df[min_col_name] > df[max_col_name]
    df.loc[swap_condition, [min_col_name, max_col_name]] = df.loc[swap_condition, [max_col_name, min_col_name]].values

    return df


# Function to correct program names
def transform_program_name(name: str) -> str:
    """
    Transforms program names to a more standardized format.

    Parameters:
    - name: Program name to transform.
    """

    start_mappings = {
        "European Master of Science in Information Systems Management": "European Master of Science in Information Systems Management",
        "Master Degree in Law and Financial Markets": "Mestrado em Direito e Mercados Financeiros",
        "Master degree program in Data Science and Advanced Analytics": "Mestrado em Ciência de Dados e Métodos Analíticos Avançados",
        "Master degree program in Geographic Information Systems and Science": "Mestrado em Ciência e Sistemas de Informação Geográfica",
        "Master degree program in Information Management": "Mestrado em Gestão de Informação",
        "Master degree program in Statistics and Information Management": "Mestrado em Estatística e Gestão de Informação",
        "Master in Data Driven Marketing": "Mestrado em Marketing Analítico (Data Driven Marketing)",
        "Master of Science in Geospatial Technologies": "Master of Science in Geospatial Technologies"
    }

    end_mappings = {
        " - Análise e Gestão de Informação": "Pós-Graduação em Análise e Gestão de Informação",
        " - Information Analysis and Management": "Pós-Graduação em Análise e Gestão de Informação",
        " - Análise e Gestão de Risco": "Pós-Graduação em Análise e Gestão de Risco",
        " - Risk Analysis and Management": "Pós-Graduação em Análise e Gestão de Risco",
        " - Business Analytics for Hospitality & Tourism": "Pós-Graduação em Business Analytics for Hospitality & Tourism",
        " - Business Intelligence": "Pós-Graduação em Business Intelligence",
        " - Business Intelligence and Analytics for Hospitality & Tourism": "Post-graduate program in Business Intelligence & Analytics for Hospitality and Tourism",
        " - Data Analytics": "Pós-Graduação em Data Analytics",
        " - Data Science for Finance": "Pós-Graduação em Data Science for Finance",
        " - Data Science for Hospitality and Tourism": "Pós-Graduação em Data Science for Hospitality and Tourism",
        " - Data Science for Marketing": "Pós-Graduação em Data Science for Marketing",
        " - Pós-Graduação em Data Science for Marketing": "Pós-Graduação em Data Science for Marketing",
        " - Digital Enterprise Management": "Pós-Graduação em Digital Enterprise Management",
        " - Digital Marketing and Analytics": "Pós-Graduação em Digital Marketing and Analytics",
        " - Transformação Digital": "Pós-Graduação em Transformação Digital",
        " - Enterprise Data Science & Analytics": "Pós-Graduação em Enterprise Data Science and Analytics",
        " - Financial Markets and Risks": "Pós-Graduação em Mercados e Riscos Financeiros",
        " - Mercados e Riscos Financeiros": "Pós-Graduação em Mercados e Riscos Financeiros",
        " - Geospatial Intelligence": "Pós-Graduação em Geospatial Intelligence",
        " - Inteligência Geoespacial": "Pós-Graduação em Geospatial Intelligence",
        " - Financial and Budgetary Management and Control": "Pós-Graduação em Gestão e Controlo Financeiro e Orçamental",
        " - Gestão de Informação e Business Intelligence na Saúde": "Pós-Graduação em Gestão de Informação e Business Intelligence na Saúde",
        " - Information Management and Business Intelligence in Healthcare": "Pós-Graduação em Gestão de Informação e Business Intelligence na Saúde",
        " - Gestão de Informações e Segurança": "Pós-Graduação em Gestão de Informações e Segurança",
        " - Intelligence Management and Security": "Pós-Graduação em Gestão de Informações e Segurança",
        " - Gestão do Conhecimento e Business Intelligence": "Pós-Graduação em Gestão do Conhecimento e Business Intelligence",
        " - Knowledge Management and Business Intelligence": "Pós-Graduação em Gestão do Conhecimento e Business Intelligence",
        " - Gestão dos Sistemas de Informação": "Pós-Graduação em Gestão dos Sistemas de Informação",
        " - Information Systems Management": "Pós-Graduação em Gestão dos Sistemas de Informação",
        " - Gestão dos Sistemas e Tecnologias de Informação": "Pós-Graduação em Gestão dos Sistemas e Tecnologias de Informação",
        " - Information Systems and Technologies Management": "Pós-Graduação em Gestão dos Sistemas e Tecnologias de Informação",
        " - Gestão e Controlo Financeiro e Orçamental": "Pós-Graduação em Gestão e Controlo Financeiro e Orçamental",
        " - Gestão e Controlo Financeiro e Orçamental na Saúde": "Pós-Graduação em Gestão e Controlo Financeiro e Orçamental na Saúde",
        " - Information Technology Product Management": "Pós-Graduação em Information Technology Product Management",
        " - Sistemas de Informação Empresariais": "Pós-Graduação em Sistemas de Informação Empresariais",
        " - Smart Cities": "Pós-Graduação em Cidades Inteligentes (Smart Cities)",
        " - Enterprise Information Systems": "Pós-Graduação em Sistemas de Informação Empresariais",
    }

    contains_mappings = {
        "Ciência dos Dados Geospaciais": "Pós-Graduação em Geospatial Data Science",
        "Geospatial Data Science": "Pós-Graduação em Geospatial Data Science",
        " - Ciência e Sistemas de Informação Geográfica": "Pós-Graduação em Ciência e Sistemas de Informação Geográfica",
        " - Science and Geographic Information System": "Pós-Graduação em Ciência e Sistemas de Informação Geográfica",
        " - Digital Transformation": "Pós-Graduação em Transformação Digital",
        " - Marketing Intelligence": "Pós-Graduação em Marketing Intelligence",
        " - Marketing Research": "Pós-Graduação em Estudos de Mercado & CRM",
        " - Sistemas Estatísticos": "Pós-Graduação em Sistemas Estatísticos",
        " - Statistical Systems": "Pós-Graduação em Sistemas Estatísticos"
    }

    for key, value in start_mappings.items():
        if name.startswith(key):
            return value
    
    for key, value in end_mappings.items():
        if name.endswith(key):
            return value
    
    for key, value in contains_mappings.items():
        if key in name:
            return value

    return name


# Function to optimize the data types of a dataset (adapted from https://stackoverflow.com/questions/57856010/automatically-optimizing-pandas-dtypes)
def auto_opt_pd_dtypes(df: imps.pd.DataFrame) -> imps.pd.DataFrame:
    """
    Automatically downcast numerical data types to the minimum possible. Will not touch other data types (datetime, str, object, etc).
    Will ignore columns that have missing values.
        
    Parameters:
    - df: Dataframe to optimize the data types.
    """
    df = df.copy()
        
    for col in df.columns:
        # integers
        if issubclass(df[col].dtypes.type, imps.numbers.Integral):
            if df[col].min() >= 0:
                df[col] = imps.pd.to_numeric(df[col], downcast = "unsigned")
            else:
                df[col] = imps.pd.to_numeric(df[col], downcast = "integer")
        # other real numbers (floats)
        elif issubclass(df[col].dtypes.type, imps.numbers.Real):
            if df[col].notna().all() and (df[col] == df[col].astype(int)).all():
                df[col] = imps.pd.to_numeric(df[col], downcast = "integer")
            else:
                df[col] = imps.pd.to_numeric(df[col], downcast = "float")
    
    return df


# Function to visualize Spearman's correlation map and print correlations above a certain threshold
def spearman_correlation(df: imps.pd.DataFrame,
                         numerical_variables: list,
                         threshold: float = 0.7) -> imps.pd.DataFrame:
    """
    Displays a heatmap with the Spearman correlation for the numerical variables in the dataset,
    and prints the pairs of variables with a correlation above a certain threshold (by default, 0.7).

    Parameters:
    - df: Dataframe to analyze.
    - numerical_variables: List of numerical variables to analyze.
    - threshold: Threshold to consider a correlation as high.
    """
    df = df.copy()

    cor = df[numerical_variables].corr("spearman")
    mask = imps.np.triu(imps.np.ones_like(cor, dtype=bool))

    imps.plt.figure(figsize = (12, 10))
    imps.sns.heatmap(data = cor,
                mask = mask,
                annot = True,
                annot_kws = {"size": 10},
                cmap = "vlag",
                vmax = 1,
                vmin = -1,
                center = 0,
                fmt = '.2')
    imps.plt.title("Spearman Correlation", y = 1.02)
    imps.plt.show()

    high_correlation_pairs = []

    for i in range(len(cor.columns)):
        for j in range(i + 1, len(cor.columns)):
            if abs(cor.iloc[i, j]) > threshold:
                high_correlation_pairs.append((cor.columns[i], cor.columns[j]))
    
    for pair in high_correlation_pairs:
        correlation_value = cor.loc[pair[0], pair[1]]
        print(f"High Correlation: {pair[0]} and {pair[1]} - Correlation: {correlation_value}")

    return cor


# Function to assess the Spearman correlation against a specified target variable
def spearman_correlation_with_target(df: imps.pd.DataFrame,
                                     target: imps.pd.Series,
                                     numerical_variables: list) -> imps.pd.DataFrame:
    """
    Computes the Spearman correlation between each numerical variable in a list and a pandas series.
    Adds a column to indicate if the correlation is greater than 0.1.
    
    Parameters:
    - df: Dataframe to analyze.
    - target: Pandas Series to compute correlation with.
    - numerical_variables: List of numerical column names to evaluate.
    """
    df = df.copy()
    results = []
    
    for col in numerical_variables:
        valid_data = df[[col]].join(target)
        x = valid_data[col]
        y = valid_data[target.name]
        
        cor, _ = imps.stats.spearmanr(x, y)
        results.append({"Variable": col,
                        "Spearman Correlation": imps.np.round(cor, 4),
                        "|Value| >= Threshold": abs(cor) > 0.1})
    
    return imps.pd.DataFrame(results)


# Function to perform an ANOVA test and return the most important features (for classification problems)
def anova_classification(df: imps.pd.DataFrame,
                         numerical_variables: list,
                         categorical_col: str = None,
                         y: imps.pd.Series = None,
                         k = "all") -> imps.pd.DataFrame:
    """
    Computes the ANOVA test between all numerical variables and a certain categorical variable (which can be in the dataset or a series)
    and returns a table containing the most correlated features with that variable.

    Parameters:
    - df: Dataframe to analyze.
    - numerical_variables: List of numerical variables to analyze.
    - categorical_col: Categorical variable to analyze.
    - y: Series containing the target variable.
    - k: Number of features to return. If "all", all features will be returned.
    """
    df = df.copy()

    if y is None:
        if categorical_col is None:
            raise ValueError("Either 'categorical_col' or 'y' must be provided.")
        y = df[categorical_col]

    anova = imps.SelectKBest(imps.f_classif, k = k)
    _ = anova.fit_transform(df[numerical_variables], y)

    support = anova.get_support()
    p_values = anova.pvalues_

    important_features = imps.pd.DataFrame({"Feature": df[numerical_variables].columns, "P-Value": imps.np.round(p_values, 6)})
    important_features["Significant"] = important_features["P-Value"] < 0.05
    important_features["Top K"] = support

    return important_features


# Function to perform an ANOVA test and return the most important features (for regression problems)
def anova_regression(df: imps.pd.DataFrame,
                     categorical_variables: list,
                     numerical_col: str = None,
                     y: imps.pd.Series = None,
                     k = "all") -> imps.pd.DataFrame:
    """
    Computes the ANOVA test between all numerical variables and a certain categorical variable (which can be in the dataset or a series)
    and returns a table containing the most correlated features with that variable.

    Parameters:
    - df: DataFrame to analyze.
    - categorical_variables: List of categorical variables to analyze.
    - numerical_col: Numerical variable to analyze.
    - y: Series containing the numerical variable (alternative to `numerical_col`).
    - k: Number of top features to return. If "all", all features will be returned.
    """
    df = df.copy()
    if y is None:
        if numerical_col is None:
            raise ValueError("Either 'numerical_col' or 'y' must be provided.")
        y = df[numerical_col]

    anova_results = []

    for cat_var in categorical_variables:
        groups = [y[df[cat_var] == level] for level in df[cat_var].unique()]
        _, p_value = imps.stats.f_oneway(*groups)
        anova_results.append({"Feature": cat_var, "P-Value": imps.np.round(p_value, 6)})

    important_features = imps.pd.DataFrame(anova_results)
    important_features["Significant"] = important_features["P-Value"] < 0.05

    return important_features


# Function to perform an ANOVA test and return the most important features (classification or regression)
def anova(prediction_task: str,
          df: imps.pd.DataFrame,
          categorical_variables: list = None,
          numerical_variables: list = None,
          categorical_col: str = None,
          numerical_col: str = None,
          y: imps.pd.Series = None,
          k = "all"):
    """
    Computes the ANOVA test between all numerical/categorical variables and a certain categorical/numerical variable (which
    can be in the dataset or a series) and returns a table containing the most correlated features with that variable.

    Parameters:
    - prediction_task: Type of prediction task to perform the test for.
    - df: DataFrame to analyze.
    - categorical_variables: List of categorical variables to analyze.
    - numerical_variables: List of numerical variables to analyze.
    - categorical_col: Categorical variable to analyze.
    - numerical_col: Numerical variable to analyze.
    - y: Series containing the numerical variable (alternative to `numerical_col`).
    - k: Number of top features to return. If "all", all features will be returned.
    """
    if prediction_task == "classification":
        return anova_classification(df, numerical_variables, categorical_col, y, k = "all")
    elif prediction_task == "regression":
        return anova_regression(df, categorical_variables, numerical_col, y, k = "all")
    else:
        raise ValueError("Invalid prediction task. Choose either 'classification' or 'regression'.")


# Function to perform a Chi-Square test and return the most important features (for classification problems)
def chi_square_test_independence_classification(df: imps.pd.DataFrame,
                                                categorical_variables: list,
                                                target_col: str = None,
                                                y: imps.pd.Series = None,
                                                alpha: float = 0.05):
    """
    Performs a Chi-Square test of independence between all categorical variables and a certain target variable (which can be a column
    of the dataset or a series), and returns the most important features for the prediction.

    Parameters:
    - df: Dataframe to analyze.
    - categorical_variables: List of categorical variables to analyze.
    - target_col: Target variable to analyze.
    - y: Series containing the target variable.
    - alpha: Significance level for the test.
    """
    df = df.copy()
    cat_df = df[categorical_variables]

    if y is None:
        if categorical_col is None:
            raise ValueError("Either 'target_col' or 'y' must be provided.")
        target = df[target_col]
    else:
        target = y

    for var in cat_df:
        dfObserved = imps.pd.crosstab(target, df[var]) 
        chi2, p, dof, expected = imps.stats.chi2_contingency(dfObserved.values)
        dfExpected = imps.pd.DataFrame(expected, columns = dfObserved.columns, index = dfObserved.index)

        if p < alpha:
            result = "{0} is IMPORTANT for Prediction".format(var)
        else:
            result = "{0} is NOT an important predictor. (Discard {0} from model)".format(var)
        print(result)


# Function to perform a Chi-Square test and return the most important features (for regression problems)
def chi_square_test_independence_regression(df: imps.pd.DataFrame,
                                            categorical_variables: list,
                                            alpha: float = 0.05):
    """
    Performs a Chi-Square test of independence between each pair of categorical variables
    in the provided list and prints whether they are correlated.

    Parameters:
    - df: Dataframe to analyze.
    - categorical_variables: List of categorical variables to analyze.
    - alpha: Significance level for the test.
    """
    df = df.copy()

    for i, var1 in enumerate(categorical_variables):
        for var2 in categorical_variables[i + 1:]:
            contingency_table = imps.pd.crosstab(df[var1], df[var2])
            _, p, _, _ = imps.stats.chi2_contingency(contingency_table.values)
            
            if p < alpha:
                print(f"{var1} is correlated with {var2}")


# Function to perform a Chi-Square test of independence, adapted to the prediction task (classification or regression)
def chi_square_test_independence(prediction_task: str,
                                 df: imps.pd.DataFrame,
                                 categorical_variables: list,
                                 alpha: float = 0.05,
                                 target_col: str = None,
                                 y: imps.pd.Series = None):
    """
    Performs a Chi-Square test of independence, adapted to the prediction task (classification or regression).

    Parameters:
    - prediction_task: Type of prediction task to perform the test for.
    - df: Dataframe to analyze.
    - categorical_variables: List of categorical variables to analyze.
    - alpha: Significance level for the test.
    - target_col: Target variable to analyze.
    - y: Series containing the target variable.
    """
    if prediction_task == "classification":
        chi_square_test_independence_classification(df, categorical_variables, target_col, y, alpha)
    elif prediction_task == "regression":
        chi_square_test_independence_regression(df, categorical_variables, alpha)
    else:
        raise ValueError("Invalid prediction task. Choose either 'classification' or 'regression'.")


# Function to perform a point-biserial correlation test and return the most important features (for classification problems)
def point_biserial_classification(df: imps.pd.DataFrame,
                                  target: imps.pd.Series,
                                  numerical_variables: list,
                                  threshold: float) -> imps.pd.DataFrame:
    """
    Computes the point-biserial correlation between all numerical variables and a certain (binary) target variable,
    and returns the most correlated features.

    Parameters:
    - df: Dataframe to analyze.
    - target: Series containing the target variable.
    - numerical_variables: List of numerical variables to analyze.
    - threshold: Threshold to consider a correlation as high.
    """
    df = df.copy()
    results = []

    for var in numerical_variables:
        corr, _ = imps.stats.pointbiserialr(df[var], target)
        results.append({"Variable": var,
                        "Value": corr,
                        "|Value| >= Threshold": abs(round(corr, 2)) >= threshold})

    return imps.pd.DataFrame(results)


# Function to perform a point-biserial correlation test and return the most important features (for regression problems)
def point_biserial_regression(df: imps.pd.DataFrame,
                              target: imps.pd.Series,
                              binary_variables: list,
                              threshold: float) -> imps.pd.DataFrame:
    """
    Computes the point-biserial correlation between all binary variables and a certain (numerical) target variable,
    and returns the most correlated features.

    Parameters:
    - df: Dataframe to analyze.
    - target: Series containing the target variable.
    - binary_variables: List of binary variables to analyze.
    - threshold: Threshold to consider a correlation as high.
    """
    df = df.copy()
    results = []

    for var in binary_variables:
        corr, _ = imps.stats.pointbiserialr(target, df[var])
        results.append({"Variable": var,
                        "Value": corr,
                        "|Value| >= Threshold": abs(round(corr, 2)) >= threshold})

    return imps.pd.DataFrame(results)


# Function to perform a point-biserial correlation test and return the most important features (classification or regression)
def point_biserial(prediction_task: str,
                   df: imps.pd.DataFrame,
                   target: imps.pd.Series,
                   threshold: float,
                   numerical_variables: list = None,
                   binary_variables: list  = None) -> imps.pd.DataFrame:
    """
    Computes the point-biserial correlation between a group of numerical/binary variables and a certain binary/numerical target variable,
    and returns the most correlated features.

    Parameters:
    - prediction_task: Type of prediction task to perform the test for.
    - df: Dataframe to analyze.
    - target: Series containing the target variable.
    - numerical_variables: List of numerical variables to analyze.
    - binary_variables: List of binary variables to analyze.
    - threshold: Threshold to consider a correlation as high.
    """
    if prediction_task == "classification":
        return point_biserial_classification(df, target, numerical_variables, threshold)
    elif prediction_task == "regression":
        return point_biserial_regression(df, target, binary_variables, threshold)
    else:
        raise ValueError("Invalid prediction task. Choose either 'classification' or 'regression'.")


# Function to perform Recursive Feature Elimination, using the passed model as the estimator of choice (classification problems)
def RFE_classification(X_train: imps.pd.DataFrame,
                       X_val: imps.pd.DataFrame,
                       y_train: imps.pd.Series,
                       y_val: imps.pd.Series,
                       model):
    """
    Performs Recursive Feature Elimination using a model passed as a parameter as the estimator of choice.
    Additionally, it plots the scores obtained on the training and validation sets for each number of features.
    """
    nof_list = imps.np.arange(1, X_train.shape[1])
    high_score = 0

    # Variable to store the optimum features
    nof = 0
    train_score_list = []
    val_score_list = []

    for n in range(len(nof_list)):
        rfe_model = model
        rfe = imps.RFE(estimator = rfe_model, n_features_to_select = nof_list[n])

        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_val_rfe = rfe.transform(X_val)

        rfe_model.fit(X_train_rfe, y_train)

        #storing results on training data
        rfe_labels_train = rfe_model.predict(X_train_rfe)
        train_score = imps.f1_score(y_train, rfe_labels_train)
        train_score_list.append(train_score)

        #storing results on validation data
        rfe_labels_val = rfe_model.predict(X_val_rfe)
        val_score = imps.f1_score(y_val, rfe_labels_val)
        val_score_list.append(val_score)

        #check best score
        if(val_score >= high_score):
            high_score = val_score
            nof = nof_list[n]

            features_to_select = imps.pd.Series(data = rfe.support_, index = X_train.columns)

    print("Optimum number of features: %d" %nof)
    print("Score with %d features: %f" % (nof, high_score))
    with imps.pd.option_context("display.max_rows", None):
        print(f"Features to select: \n{features_to_select}")

    imps.plt.plot(list(range(1, X_train.shape[1])), train_score_list, label = "Score on Training Set", color = "#BFD62F")
    imps.plt.plot(list(range(1, X_train.shape[1])), val_score_list, label = "Score on Validation Set", color = "#5C666C")
    imps.plt.xlabel("Maximum Depth")
    imps.plt.ylabel("Score")
    imps.plt.legend()
    imps.plt.show()


# Function to perform Recursive Feature Elimination, using the passed model as the estimator of choice (regression problems)
def RFE_regression(X_train: imps.pd.DataFrame,
                   X_val: imps.pd.DataFrame,
                   y_train: imps.pd.Series,
                   y_val: imps.pd.Series,
                   model):
    """
    Performs Recursive Feature Elimination using a model passed as a parameter as the estimator of choice.
    Additionally, it plots the RMSE obtained on the training and validation sets for each number of features.
    
    Parameters:
    - X_train: Training feature set.
    - X_val: Validation feature set.
    - y_train: Training target values.
    - y_val: Validation target values.
    - model: Estimator to use for feature elimination and prediction.
    """
    nof_list = imps.np.arange(1, X_train.shape[1])
    low_score = float("inf")

    # Variable to store the optimum features
    nof = 0
    train_rmse_list = []
    val_rmse_list = []

    for n in range(len(nof_list)):
        rfe_model = model
        rfe = imps.RFE(estimator = rfe_model, n_features_to_select = nof_list[n])

        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_val_rfe = rfe.transform(X_val)

        rfe_model.fit(X_train_rfe, y_train)

        #storing results on training data
        rfe_preds_train = rfe_model.predict(X_train_rfe)
        train_rmse = imps.np.sqrt(imps.mean_squared_error(y_train, rfe_preds_train))
        train_rmse_list.append(train_rmse)

        #storing results on validation data
        rfe_preds_val = rfe_model.predict(X_val_rfe)
        val_rmse = imps.np.sqrt(imps.mean_squared_error(y_val, rfe_preds_val))
        val_rmse_list.append(val_rmse)

        #check best score
        if val_rmse <= low_score:
            low_score = val_rmse
            nof = nof_list[n]
            features_to_select = imps.pd.Series(data=rfe.support_, index=X_train.columns)

    print("Optimum number of features: %d" % nof)
    print("Lowest RMSE with %d features: %f" % (nof, low_score))
    with imps.pd.option_context("display.max_rows", None):
        print(f"Features to select: \n{features_to_select}")

    imps.plt.plot(list(range(1, X_train.shape[1])), train_rmse_list, label = "RMSE on Training Set", color = "#BFD62F")
    imps.plt.plot(list(range(1, X_train.shape[1])), val_rmse_list, label = "RMSE on Validation Set", color = "#5C666C")
    imps.plt.xlabel("Number of Features Selected")
    imps.plt.ylabel("RMSE")
    imps.plt.legend()
    imps.plt.show()


# Function to perform Recursive Feature Elimination, using the passed model as the estimator of choice (classification or regression)
def RFE(prediction_task: str,
        X_train: imps.pd.DataFrame,
        X_val: imps.pd.DataFrame,
        y_train: imps.pd.Series,
        y_val: imps.pd.Series,
        model):
    """
    Performs Recursive Feature Elimination using a model passed as a parameter as the estimator of choice.
    Additionally, it plots the RMSE obtained on the training and validation sets for each number of features.
    
    Parameters:
    - prediction_task: Type of prediction task to perform the test for.
    - X_train: Training feature set.
    - X_val: Validation feature set.
    - y_train: Training target values.
    - y_val: Validation target values.
    - model: Estimator to use for feature elimination and prediction.
    """
    if prediction_task == "classification":
        RFE_classification(X_train, X_val, y_train, y_val, model)
    elif prediction_task == "regression":
        RFE_regression(X_train, X_val, y_train, y_val, model)
    else:
        raise ValueError("Invalid prediction task. Choose either 'classification' or 'regression'.")


# Function to return the most important features when computing a default random forest (classification or regression)
def random_forest_feature_selection(df: imps.pd.DataFrame,
                                    target: imps.pd.Series,
                                    threshold: float,
                                    rf_model):
    """
    Computes the feature importance for a default Random Forest model and plots the results in a horizontal bar chart.

    Parameters:
    - df: Dataframe to analyze.
    - target: Target variable to predict.
    - threshold: Threshold to consider a feature as important.
    """
    df = df.copy()

    model = rf_model
    model.fit(df, target)
    importances = model.feature_importances_
    indices = imps.np.argsort(importances)[::-1]

    feature_importances = imps.pd.DataFrame({"Feature": df.columns[indices],
                                             "Importance": importances[indices]})

    fig = imps.px.bar(feature_importances, x = "Importance", y = "Feature", orientation = "h", color_discrete_sequence = ["#5C666C"],
                      title = "Feature importance (for a default Random Forest)")
    fig.add_vline(x = threshold, line_dash = "dash", line_color = "black")
    fig.update_layout(title_x = 0.5, showlegend = False, height = len(feature_importances) * 20)
    fig.show()


# Function to return the most important features detected by the LASSO method
def lasso(df: imps.pd.DataFrame,
          target: imps.pd.Series,
          threshold: float):
    """
    Computes the feature importance using the LASSO method.

    Parameters:
    - df: Dataframe to analyze.
    - target: Target variable to predict.
    - threshold: Threshold to consider a feature as important.
    """
    df = df.copy()

    reg = imps.LassoCV(random_state = 92)
    reg.fit(df, target)
    coef = imps.pd.Series(data = reg.coef_, index = df.columns)
    imp_coef = coef.sort_values()
    imp_coef_df = imps.pd.DataFrame({"Feature": imp_coef.index, "Importance": imp_coef.values})

    print("LASSO picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables.")

    fig = imps.px.bar(imp_coef_df, x = "Importance", y = "Feature", orientation = "h", color_discrete_sequence = ["#5C666C"],
                      title = "Feature Importance using the Lasso Method")
    fig.add_vline(x = threshold, line_dash = "dash", line_color = "black")
    fig.add_vline(x = -threshold, line_dash = "dash", line_color = "black")
    fig.update_layout(title_x = 0.5, xaxis_title = "Coefficient", yaxis_title = "Feature", height = len(imp_coef_df) * 20)
    fig.show()


# Function to highlight the background of a cell based on its value
def highlight_cell(val, column):
    """
    Highlights the background of a cell based on its value.

    Parameters:
    - val: Value of the cell.
    - column: Column where the cell is located.
    """
    if 'Result' in column:
        return 'background-color: #FFFF66' if val == 'Keep' else ''
    elif val == "Keep":
        return 'background-color: #99FF99'
    elif val == "Discard":
        return 'background-color: #FF8989'
    return ""


# Function to evaluate the performance of a given model using different oversampling techniques
def evaluate_oversampling_techniques(X_train: imps.pd.DataFrame,
                                     X_val: imps.pd.DataFrame,
                                     y_train: imps.pd.Series,
                                     y_val: imps.pd.Series,
                                     model,
                                     categorical_variables: list,
                                     neighbors = 5,
                                     random_seed = None) -> imps.pd.DataFrame:
    """
    Compares different oversampling techniques to improve the performance of a given model.

    Parameters:
    - X_train: Training data.
    - X_val: Validation data.
    - y_train: Training target.
    - y_val: Validation target.
    - model: Model to evaluate.
    - categorical_variables: List of categorical variables.
    - neighbors: Number of neighbors to consider (where applicable).
    - random_seed: Random seed for reproducibility.
    """
    if random_seed is not None:
        imps.np.random.seed(random_seed)
    
    samplers = [
        imps.SMOTE(random_state = 92, k_neighbors = neighbors),
        imps.ADASYN(random_state = 92, n_neighbors = neighbors),
        imps.RandomOverSampler(random_state = 92),
        imps.SMOTENC(random_state = 92, categorical_features = categorical_variables, k_neighbors = neighbors),
        imps.BorderlineSMOTE(random_state = 92, k_neighbors = neighbors, m_neighbors = neighbors * 2),
        imps.KMeansSMOTE(random_state = 92, k_neighbors = neighbors, cluster_balance_threshold = 0.05),
        imps.SVMSMOTE(random_state = 92, k_neighbors = neighbors, m_neighbors = neighbors * 2)
    ]

    results = []
    for sampler in samplers:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        classifier = model
        classifier.fit(X_resampled, y_resampled)
        y_pred = classifier.predict(X_val)
        f1_score = imps.f1_score(y_val, y_pred)
        precision_score = imps.precision_score(y_val, y_pred)
        recall_score = imps.recall_score(y_val, y_pred)

        results.append({"Sampler": sampler.__class__.__name__, "F1 score": f1_score, "Precision": precision_score, "Recall": recall_score})

    results_df = imps.pd.DataFrame(results)
    results_df = results_df.sort_values(by = "F1 score", ascending = False)

    best_sampler_row = results_df.loc[results_df["F1 score"].idxmax()]
    best_sampler = best_sampler_row["Sampler"]
    best_f1_score = best_sampler_row["F1 score"]

    print(f"Best sampling technique: {best_sampler} with F1 score: {best_f1_score}")
    return results_df


# Function to run a grid search on a passed model, train it on the best identified parameters, and save the model to a pickle file
def run_model_classification(model,
                             param_grid: dict,
                             X_train: imps.pd.DataFrame,
                             y_train: imps.pd.Series,
                             X_val: imps.pd.DataFrame,
                             y_val: imps.pd.Series,
                             scoring = "f1",
                             save_model = False,
                             model_name = None):
    """
    Performs a grid search for the given model, scores it using the specified metric, and evaluates it on the validation dataset,
    calculating the accuracy, precision, recall, F1 score, and AUROC (if applicable).
    It also saves the model to a pickle file (optional).

    Parameters:
    - model: The model to optimize and train.
    - param_grid: The parameter grid for GridSearchCV.
    - X_train: Training features.
    - y_train: Training target variable.
    - X_val: Validation features.
    - y_val: Validation target variable.
    - scoring: The metric used to evaluate model performance during GridSearchCV.
    - save_model: Option to save the model into a pickle file.
    - model_name: Pickle file name.
    """
    grid_search = imps.GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, n_jobs = -1, cv = 5, verbose = 1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, "predict_proba") else None

    accuracy = imps.np.round(imps.accuracy_score(y_val, y_pred), 4)
    precision = imps.np.round(imps.precision_score(y_val, y_pred), 4)
    recall = imps.np.round(imps.recall_score(y_val, y_pred), 4)
    f1 = imps.np.round(imps.f1_score(y_val, y_pred), 4)
    roc_auc = imps.np.round(imps.roc_auc_score(y_val, y_pred_proba), 4) if y_pred_proba is not None else "-"

    print("\nClassification Report:\n", imps.classification_report(y_val, y_pred))
    print("Accuracy:", accuracy, "| Precision:", precision, "| Recall:", recall, "| F1 score:", f1, "| AUROC:", roc_auc)

    if save_model is True:
        model_dir = "admissions"
        imps.os.makedirs(model_dir, exist_ok = True)
        model_filename = imps.os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_filename, "wb") as file:
            imps.pickle.dump(best_model, file)

    return best_params, y_pred, accuracy, precision, recall, f1, roc_auc


# Function to customize the background of a cell in a table, highlighting the maximum value for each row
def highlight_max_column(df: imps.pd.DataFrame):
    """
    Returns the dataframe, highlighting the maximum value for each row
    """
    return imps.df.style.apply(lambda col: ["background-color: lightgreen" if value == col.max() else "" for value in col], axis = 0)