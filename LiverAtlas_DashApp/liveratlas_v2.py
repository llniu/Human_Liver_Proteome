import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import base64
import ast
import umap
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px


# Import figures and datasets
image1 = 'images/webpage_image.png'
image2= 'images/studydesign.png'
image3= 'images/cpr_logo.png'
image4= 'images/ku_logo.png'
image5= 'images/Liverdisease.png'


## Statistics data
statistics = pd.read_csv('dataset/statistics.csv').iloc[:, 1:]
## Protein MS evidence summary data
data_summary = pd.read_csv('dataset/data_summary.csv', index_col = 'Genename_ProteinID')
## Peptide evidence dataset
data_peptide = pd.read_csv('dataset/data_peptide.csv', index_col = 'Genename_ProteinID')
data_peptide_cols = data_peptide.columns
## Atlas dataset before and after imputation, for umap
dataset_atlas_imputed = pd.read_csv('dataset/atlas_34_log10_imputed.csv').iloc[:, 1:]
dataset_atlas_log10 = pd.read_csv('dataset/atlas_34_log10.csv').set_index('Protein ID').astype(np.number)
dataset_copyno = pd.read_csv('dataset/atlas_34_copyno.csv').set_index('Protein ID').astype(np.number)
## Cell type marker dataset, for heatmap
df_unique = pd.read_csv('dataset/107specific genes_Ab_tissue_disease.csv')
## Patient tissue ttest data, for volcano plot
dataset = pd.read_csv('dataset/Volcano_atlas_modified222.csv')
## NAFLD plasma cirrhosis ttest data, for volcano plot
df_plasma = pd.read_csv('dataset/NAFLD_plasma_dataset_520_log2_ttest_cirrhosis.csv')

## Import ID annotation files
kegg = pd.read_csv('dataset/pathway_dict.csv')
ID_annotation = pd.read_csv('annotations/Perseus_annotation_file.csv')
IDmapping_ProteinID_to_GeneProteinID = dict(zip(ID_annotation['Protein ID'], ID_annotation['Genename_ProteinID']))
experiment_annotation = pd.read_csv('annotations/Experiment_annotation_file.csv')
grouping1 = dict(zip(experiment_annotation['Label4'], experiment_annotation['Label6']))


################################################################################
## Prepare kegg pathway associated gene neames dict
kegg_dict = dict(zip(kegg.pathway, [ast.literal_eval(a) for a in kegg['gene names']]))
options_kegg = [{"label": i, "value":i} for i in kegg_dict.keys()]

## Prepare images
def png_encode(image_png):
    png_base64 = base64.b64encode(open(image_png, 'rb').read()).decode('ascii')
    return (png_base64)
image1_base64 = png_encode(image1)
image2_base64 = png_encode(image2)
image3_base64 = png_encode(image3)
image4_base64 = png_encode(image4)
image5_base64 = png_encode(image5)

## Prepare umap in sample overview
options_umap = [{'label':i, 'value':i} for i in dataset_atlas_imputed['index'].unique()]

## Prepare cell type marker data
df_unique = df_unique.set_index('Gene symbol')
df_unique = df_unique.sort_values(by = ['hHEP', 'hHSC', 'hKC', 'hLSEC'], ascending = False)
df_unique_hm = df_unique.iloc[:, 1:13]

dumb_a = ['hHEP'] *3 + ['hHSC'] *3 + ['hKC'] *3 + ['hLSEC'] *3
dumb_b = dict(zip(df_unique.columns[1:13], dumb_a))
df_unique_tb = df_unique.rename(mapper = dumb_b, axis=1).reset_index()
options_celltype_markers = [{'label': i, 'value': i} for i in ['hHEP', 'hHSC', 'hKC', 'hLSEC']]

## Prepare data for proteome depth
grouping2 = dict(zip(experiment_annotation['Label4'], experiment_annotation['Label3']))
id_no = dataset_atlas_log10.groupby(by = grouping2, axis = 1).median().count()
id_no['total'] = dataset_atlas_log10.shape[0]

## Prepare data for volcano plot in tissue
df = dataset.copy()
df_sig = df[df['Significant'] == '+']
x = dataset['Log2 fold change']
y = dataset['-Log p-value']
dt_columns = ['Leading Gene name', 'Leading Protein ID', 'Protein name', '-Log p-value', 'Log2 fold change']
options_GeneNames_volcano = [{'label': i, 'value' : i} for i in df['Leading Gene name'].dropna()]
## Prepare data for expression across tissus/cell types
dataset_atlas = np.power(dataset_atlas_log10, 10)
dataset_atlas_median = dataset_atlas.groupby(by = grouping1, axis = 1).median()
dataset_atlas_median['Genename_ProteinID'] = dataset_atlas_median.index.map(IDmapping_ProteinID_to_GeneProteinID)
reorder = ['Liver', 'PorV', 'HepA', 'hHEP', 'hKC', 'hHSC', 'hLSEC', 'Hep G2', 'LX2', 'hHSC TWNT4', 'hLSEC SK Hep1']
dataset_atlas_median = dataset_atlas_median.set_index('Genename_ProteinID')[reorder]
lfq_median_log10 = np.log10(dataset_atlas_median)
df_lfq = dataset_atlas_median.copy().fillna(1)

copyno_median = dataset_copyno.groupby(by = grouping1, axis = 1).median()
copyno_median['Genename_ProteinID'] = copyno_median.index.map(IDmapping_ProteinID_to_GeneProteinID)
copyno_median = copyno_median.set_index('Genename_ProteinID')[reorder].fillna(1)
df_copyno = copyno_median.copy()
options_GeneNames = [{'label':i, 'value':i} for i in df_lfq.index]

## Prepare data for abundace rank plotly
def prepare_data_for_rank(celltype):
    df_new = lfq_median_log10.copy().reset_index()
    df_new = df_new[['Genename_ProteinID', celltype]].dropna(subset = [celltype])
    df_new = df_new.sort_values(by = celltype, ascending = False).reset_index(drop = True)
    df_new['Abundance rank']=df_new.index + 1
    return df_new

df_rank_hep = prepare_data_for_rank('hHEP')
df_rank_lsec = prepare_data_for_rank('hLSEC')
df_rank_hsc = prepare_data_for_rank('hHSC')
df_rank_kc = prepare_data_for_rank('hKC')

################################################################################
#building Dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#Styles
style_headerh1 = {'backgroundColor':'#243E58', 'color':'snow', 'textAlign':'center',
                    'height':'120px', 'line-height': '120px', 'border': '2px solid white',
                    'font-style':'normal', 'font-family':'Copperplate', 'first-letter':{'color':'red'}
                    }
style_headerh2 = {'color':'black', 'textAlign':'center', 'fontSize':'8'}
style_graph_4panel = {'display': 'inline-block', 'width':'24%', 'textAlign':'center'}
style_graph_2panel = {'display': 'inline-block', 'width' : '48%', 'textAlign':'center'}

tab_style = {'height': '40px', 'border': '1px solid black',
    'padding': '6px',}
tab_style1 = {'height': '80px', 'border': '1px solid black', 'backgroundColor':'whitesmoke',
    'line-height': '40px', 'textAlign':'center', 'font-size':'120%', 'color':'#243E58'}
selected_tab_style = {'height': '80px', 'border': 'none', 'backgroundColor':'white',
    'line-height': '40px', 'textAlign':'center', 'font-size':'120%', 'color':'#243E58'}

style_cell_1 = {'minWidth': '10px', 'width': '10px', 'maxWidth': '10px',
'border': '1px solid black', 'backgroundColor':'whitesmoke', 'color':'black',
'overflow':'hidden', 'textOverflow':'ellipsis', 'font-size':'110%', 'textAlign':'left'}

style_cell_2 = {'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
'border': '1px solid black', 'overflow':'ellipsis','backgroundColor':'whitesmoke', 'color':'black'}

style_userinput= {'font-size':'120%', 'color':'darkred'}
table_header_color='#12395d'
################################################################################
################################################################################
app.layout = html.Div([
    html.H1('HUMAN LIVER PROTEOME DATABASE', style = style_headerh1),
    html.Div([
        dcc.Tabs(id = 'tabs', value = '1', children = [
            dcc.Tab(id = 'tab1', label = 'About', value = '1', children = [
                html.Div([
                    html.Br(),
                    html.Br(),
                    html.Img(src='data:image/png;base64,{}'.format(image1_base64),
                            style={
                                'height':'60%',
                                'width' :'60%'})
                ], style = {'width': '60%', 'textAlign':'center', 'padding-left':'20%'}),
                html.Br(),
                html.Div([
                    html.H2('About the database', style = style_headerh2),
                    html.Div([
                        html.Div([
                            html.H5('Content:'),
                            dcc.Markdown('''1. The human liver proteome database (HLPD) provides probably the largest
                                    cell-type resolved human liver proteome data by MS-based proteomics.
                                    It hosts quantative information about proteins across four major
                                    liver cell types: **hepatocytes**, **sinusodial endothelial cells**,
                                    **hepatic stellate cells**, and **Kupffer cells** (label free quantification and protein copy
                                    number estimation).''', style = {'textAlign':'justify', 'before':{'content':'-'}}),
                            dcc.Markdown('''2. In addition to in-depth proteomes, HLPD includes proteome data of patients
                            with **cirrhosis** and **NASH**, presenting dysregulated proteome at both protein and pathway level
                            in **liver** and **plasma**, which can indicate areas of potentical therapeutic intervention.''', style = {'textAlign':'justify'}),
                            dcc.Markdown('''3. As an ongoing effort [in the study of liver disease](https://link.springer.com/chapter/10.1007/978-3-319-98890-0_11),
                            HLPD will continuousy include other in-house generated patho-physiological proteomics data.''',
                            style = {'textAlign':'justify'}
                            ),
                            html.Br(),
                            html.H5('Workflow:'),
                            html.Div([
                                html.Img(src='data:image/png;base64,{}'.format(image2_base64),style={'height':'60%','width' :'60%'}),
                            ], style = {'textAlign':'center'}),
                            html.H5('References:'),
                            dcc.Markdown('1. [A cell-type resolved human liver proteome atlas](https://www.biorxiv.org/content/10.1101/2022.01.28.478194v1.full)'),
                            dcc.Markdown('2. [Plasma proteome dataset in NAFLD and cirrhosis](https://www.embopress.org/doi/full/10.15252/msb.20188793)'),
                            html.Br(),
                            html.H5('Funding agencies:'),
                            dcc.Markdown('[the Novo Nordisk Foundation](https://novonordiskfonden.dk/en/)'),
                            html.Br(),
                        ], style = {'width':'80%', 'padding-left':'10%'})
                    ]),
                ], style={'backgroundColor':'whitesmoke'}),
                html.Div([
                    html.Div([
                        html.Br(),
                        html.H2('Beyond the database', style = style_headerh2),
                        html.Br(),
                        dcc.Markdown('**Towards Translational Research**'),
                        dcc.Markdown('**Understand pathogenesis & progression of liver diseases**'),
                        dcc.Markdown('**Discovery & validation of biomarkers**'),
                        html.Br(),
                        html.Img(src='data:image/png;base64,{}'.format(image5_base64),
                        style = {'height':'50%', 'width':'50%'}),
                    ])
                ], style={'textAlign':'center'}),
                html.Div([
                    html.Div([
                        html.Br(),
                        dcc.Markdown('Laboratories of Prof. Matthias Mann at [University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/) and [Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann)'),
                        dcc.Markdown('Laboratory of Prof. Jonel Trebicka, Goethe-University Frankfurt'),
                        dcc.Markdown('[The MicrobLiver Consortium](https://www.sdu.dk/en/flash/projects/microbliver)'),
                        dcc.Markdown('If you have any feedback or request, please contact [Lili Niu](https://www.rasmussenlab.org/people/lili_niu) **lili.niu@cpr.ku.dk**'),
                    ], style = {'width':'50%', 'display':'inline-block', 'padding-left':'25%','textAlign':'center'}),
                ], style = {'backgroundColor':'whitesmoke'}),
                html.Br(),
                html.Div([
                    html.Div([
                        html.A([
                            html.Img(src='data:image/png;base64,{}'.format(image3_base64),
                            style = {'height':'20%', 'width':'20%'})
                        ], href = 'https://www.cpr.ku.dk'),
                    ], style = {'width':'49%', 'display':'inline-block', 'textAlign':'left'}),
                    html.Div([
                        html.A([
                            html.Img(src='data:image/png;base64,{}'.format(image4_base64),
                            style = {'height':'30%', 'width':'30%'})
                        ], href = 'https://www.ku.dk/english/'),
                    ], style = {'width':'49%', 'display':'inline-block', 'float':'right', 'textAlign':'right'})
                ], style = {'backgroundColor':'whitesmoke'}),
            ], style = tab_style1, selected_style = selected_tab_style),
            dcc.Tab(label = 'Query', value = '4', children = [
                html.Br(),
                dcc.Tabs([
                    dcc.Tab(label = 'Abundance across cell types', value = '4.1', children = [
                        html.Div([
                                html.Div([
                                    html.P('Type in a Gene name or Uniprot ID):', style = style_userinput),
                                    dcc.Dropdown(id = 'BarPlotAcrossTissue_input',
                                                value = 'ALDOB__P05062',
                                                options = options_GeneNames,
                                                style = {'width':'50%'}),
                                    html.Div([
                                        dcc.Graph(id = 'BarPlotAcrossTissue'),
                                    ], style=style_graph_2panel),
                                    html.Div([
                                        dcc.Graph(id = 'BarPlotAcrossTissue_copyno'),
                                    ], style=style_graph_2panel),
                                ], style = {'width': '68%', 'display': 'inline-block', 'padding-left': '10px',
                                'padding-top':'10px'}),
                                html.Div([
                                    html.Br(),
                                    html.Br(),
                                    html.P(id = 'AbundanceAcrossTissue_show'),
                                    dash_table.DataTable(
                                        style_header = {'fontWeight':'bold', 'font-size':'120%', 'backgroundColor':table_header_color, 'color':'white'},
                                        style_cell = style_cell_1,
                                        id = 'AbundanceAcrossCelltype_dt',
                                        columns = [{'name':i, 'id':i} for i in ['entry', 'LFQ', 'Protein copy number']]
                                    )
                                ], style = {'width': '28%', 'display':'inline-block', 'float':'right',
                                'padding-right': '20px'})
                        ]),
                        html.Div([
                            dcc.Markdown("**Link to Uniprot:**", style = style_userinput),
                            dcc.Markdown(id = 'protein_Uniprot_link'),
                            dcc.Markdown('**Link to Human Protein Atlas:**', style = style_userinput),
                            dcc.Markdown(id = 'protein_HPA_link'),
                            dcc.Markdown('**Summary of MS evidence:**', style = style_userinput),
                            html.Div([
                                dash_table.DataTable(id = 'protein_summary_table',
                                    columns = [{'name':i, 'id':i} for i in ['entry', 'value']],
                                    style_header={'font-size':'120%','fontWeight': 'bold', 'backgroundColor':table_header_color, 'color':'white'},
                                    style_table = {'textAlign':'left'},
                                    style_cell = {'border': '1px solid black', 'color':'black',
                                'font-size':'120%','textAlign':'left'}),
                            ]),
                            html.Br(),
                            dcc.Markdown('**Identified peptides:**', style = style_userinput),
                            html.Div([
                                dash_table.DataTable(id = 'peptide_summary_table',
                                    columns = [{'name':i, 'id':i} for i in data_peptide_cols],
                                    sort_action = 'native',
                                    style_header = {'font-size':'120%', 'fontWeight':'bold', 'backgroundColor':table_header_color, 'color':'white'},
                                    style_table= {'textAlign':'left', 'overflowY': 'scroll',},
                                    style_data={'height':'auto', 'whiteSpace':'normal'},
                                    style_cell= {'border':'1px solid black', 'color':'black',
                                    'font-size':'120%', 'textAlign':'left', 'textOverflow':'hidden'})
                            ]),
                        ], style = {'width':'90%', 'display':'block-inline', 'padding-left':'5%'})
                    ], style = tab_style, selected_style = selected_tab_style),
                    dcc.Tab(label = 'Abundance rank', value = '4.2', children = [
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.P('Type in a Gene name or Uniprot ID: (the query may take a few seconds)', style=style_userinput),
                                    dcc.Dropdown(
                                        id = 'ProteinID_abundance_rank', options = options_GeneNames,
                                        value = 'ALDOB__P05062')],
                                )
                                ], style = {'padding-left':'10px', 'width': '29%', 'display':'inline-block',
                                'padding-top':'10px'}
                            ),
                            html.Div([
                                html.Div([
                                    dcc.Graph(id = 'AbundanceRankPlot_HEP')
                                ], style = style_graph_4panel),

                                html.Div([
                                    dcc.Graph(id = 'AbundanceRankPlot_LSEC')
                                ], style = style_graph_4panel),

                                html.Div([
                                    dcc.Graph(id = 'AbundanceRankPlot_KC')
                                ], style = style_graph_4panel),

                                html.Div([
                                    dcc.Graph(id = 'AbundanceRankPlot_HSC')
                                ], style = style_graph_4panel)
                            ])
                        ])
                    ], style = tab_style, selected_style = selected_tab_style),
                    dcc.Tab(label = 'Cell type markers', value = '4.3', children = [
                        html.Div([
                            html.Div([
                                dcc.Graph(id = 'cell_type_marker_heatmap',
                                        figure = {'data': [go.Heatmap(x=df_unique_hm.columns,
                                                            y=df_unique_hm.index,
                                                            z=df_unique_hm.values,
                                                            colorscale = 'Cividis')],
                                                'layout': {'margin': dict(l=100, r=100, t=60, b=80)}
                                                    })
                            ], style = style_graph_2panel),
                            html.Div([
                                html.P('Select a cell type here:', style= style_userinput),
                                dcc.Dropdown(id = 'cell_type_marker_input',
                                            value = 'hHEP',
                                            options = options_celltype_markers),
                                dash_table.DataTable(
                                    id = 'cell_type_marker_table',
                                    columns = [{'name':i, 'id':i} for i in df_unique_tb.columns],
                                    sort_action = 'native',
                                    fixed_rows={ 'headers': True, 'data': 0 },
                                    style_header={'fontWeight': 'bold', 'backgroundColor':table_header_color, 'color':'white'},
                                    #style_cell=style_cell_2,
                                    style_cell={'border':'1px solid black', 'color':'black',
                                    'font-size':'120%', 'textAlign':'left', 'textOverflow':'hidden'},
                                    style_data={'whiteSpace':'normal', 'height':'auto'},
                                    style_table = {
                                        'overflowY': 'scroll',
                                        'textOverflow':'ellipsis',
                                    }
                                ),
                                dcc.Markdown('Note value of "5" means not detected. Tissue category information is from the [Human Protein Atlas](https://www.proteinatlas.org)'),
                            ], style = {'display': 'inline-block', 'width' : '46%', 'float':'right',
                             'padding-right':'40px', 'padding-top':'10px'})
                        ])
                    ], style = tab_style, selected_style = selected_tab_style)
                ])
            ], style = tab_style1, selected_style = selected_tab_style),
            dcc.Tab(label = 'Patient data', value = '5', children = [
                html.Br(),
                dcc.Tabs([
                    dcc.Tab(label = 'Cirrhosis', value = '5.1', children = [
                        html.Div([
                            html.Div([
                                html.P('Select a protein here:', style = style_userinput),
                                dcc.Dropdown(
                                    id = 'ProteinID', options = options_GeneNames_volcano,
                                    value = 'PIGR'
                                ),
                                html.P('Select a Kegg pathway here', style= style_userinput),
                                dcc.Dropdown(
                                    id = 'pathway_inference',
                                    options = options_kegg,
                                    value = 'ECM-receptor interaction'
                                )
                            ], style = style_graph_2panel)
                        ],style = {
                            'textAlign': 'center',
                            'padding': '10px 5px'
                        }),
                        html.Div([
                            dcc.Graph(id = 'VolcanoPlot')
                        ], style= {'width':'48%', 'display':'inline-block', 'textAlign':'center', 'padding-left':'2%'}),
                        html.Div([
                            dcc.Graph(id = 'VolcanoPlot_plasma_cirrhosis')
                        ], style = {'width':'48%', 'display':'inline-block', 'textAlign':'center', 'padding-right':'2%'}),
                        html.Div([
                            html.Div([
                                html.Div([
                                    dcc.Markdown('Log2 fold change range'),
                                    dcc.RangeSlider(
                                        id = 'fold-change-range-slider',
                                        min = -3,
                                        max = 3,
                                        step = 1,
                                        marks = {str(i) : {'label': str(i), 'style': {'color': '#77b0b1'}}for i in np.arange(-3, 4, 1)},
                                        value = [-1, 1]
                                        ),
                                ], style = {'padding-left':'20%', 'padding-right':'20%', 'textAlign':'center'})
                            ], style = {'width':'48%', 'display':'inline-block', 'textAlign':'center'}),

                            html.Div([
                                html.Div([
                                    dcc.Markdown('Log2 fold change range'),
                                    dcc.RangeSlider(
                                        id = 'fold-change-range-slider-plasma',
                                        min = -3,
                                        max = 3,
                                        step = 1,
                                        value = [0, 0],
                                        marks = {str(i) : {'label': str(i), 'style': {'color': '#77b0b1'}}for i in np.arange(-3, 4, 1)},
                                        ),
                                ], style = {'padding-left':'11%', 'padding-right':'20%', 'textAlign':'center'})
                            ], style = {'display': 'inline-block',
                                            'width' : '45%',
                                            'float':'right'})
                        ], style = {
                            'padding': '10px 10px 20px 20px'}),
                        html.Br(),
                        html.Br(),
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.P('Selected protein:', style = {'font-size':'120%', 'color':'darkred'}),
                                    dash_table.DataTable(
                                        id = 'volcano_table',
                                        columns = [{'name':i, 'id':i} for i in dt_columns],
                                        sort_action = 'native',
                                        style_header={'fontWeight': 'bold'},
                                        style_cell=style_cell_1,
                                        style_table = {
                                            'maxHeight': '300px',
                                            'overflowY': 'scroll',}
                                    ),
                                    html.Br(),
                                    html.P('Selected pathway:', style = {'font-size':'120%', 'color':'darkred'}),
                                    dash_table.DataTable(
                                        id = 'volcano_table_pathway',
                                        columns = [{'name':i, 'id':i} for i in dt_columns],
                                        sort_action = 'native',
                                        fixed_rows={ 'headers': True, 'data': 0 },
                                        style_header={'fontWeight': 'bold'},
                                        style_cell={
                                            'font-size':'115%',
                                            'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
                                            'overflow': 'hidden',
                                            'textOverflow': 'ellipsis',
                                            'backgroundColor': 'white',
                                            'color': 'black',
                                            'border': '1px solid grey',
                                            'textAlign':'left'},
                                        style_table = {
                                            'maxHeight': '300px',
                                            'overflowY': 'scroll'}
                                    )
                                ], style={'width':'70%', 'textAlign':'left','padding-left':'15%', 'display':'block-inline'}),
                            ])
                        ])
                    ], style = tab_style, selected_style = selected_tab_style),
                    dcc.Tab(label = 'NASH', value = '5.2', children = [
                        html.Div([
                            html.Br(),
                            html.Br(),
                            html.P('Coming soon.', style = {'textAlign':'center', 'color':'darkred'})
                        ])
                    ], style = tab_style, selected_style = selected_tab_style),
                    dcc.Tab(label = 'NAFL', value = '5.3', children = [
                        html.Div([
                            html.Br(),
                            html.Br(),
                            html.P('Coming soon.', style = {'textAlign':'center', 'color':'darkred'})
                        ])
                    ], style = tab_style, selected_style = selected_tab_style),
                ])
            ], style = tab_style1, selected_style = selected_tab_style)
        ])
    ]
    )
], style = {'width': '100%'})
################################################################################
################################################################################
@app.callback(
    [Output('BarPlotAcrossTissue', 'figure'),
     Output('BarPlotAcrossTissue_copyno', 'figure'),
     Output('AbundanceAcrossCelltype_dt', 'data'),
     Output('AbundanceAcrossTissue_show', 'children'),
     Output('protein_Uniprot_link', 'children'),
     Output('protein_HPA_link', 'children'),
     Output('protein_summary_table','data'),
     Output('peptide_summary_table', 'data')],
    [Input('BarPlotAcrossTissue_input', 'value')]
)

def update_AcrossTissueGraph(input_value):
    filtered_lfq = df_lfq.loc[input_value]
    filtered_copyno = df_copyno.loc[input_value]

    data_dt = pd.concat([filtered_lfq, filtered_copyno], axis = 1).reset_index()
    data_dt = data_dt.round(0)
    data_dt.columns = ['entry', 'LFQ', 'Protein copy number']

    types=['tissues']*3 + ['cells']*4 + ['cell lines']*4
    colors={'tissues':'#1d5c96', 'cells':'#7db0de', 'cell lines':'#12395d'}
    figure = px.bar(x=filtered_lfq.index, y=filtered_lfq, text_auto='.2s',
                    color=types, color_discrete_map=colors, log_y=True, title='Label free quantification',
                    labels={'x':'', 'y':'LFQ intensity'})
    figure.update_traces(width=0.5)

    figure_cpno = px.bar(x=filtered_copyno.index, y=filtered_copyno, text_auto='.2s',
                    color=types, color_discrete_map=colors, log_y=True, title='Protein copy number',
                    labels={'x':'', 'y':'Protein copy number'})
    figure_cpno.update_traces(width=0.5)

    data = data_dt.to_dict('records')
    table_header = 'You have selected: {}'.format(input_value)

    proteinID = input_value.split('__')[1]
    geneName = input_value.split('__')[0]
    uniprot_link = '[Protein {}:](https://www.uniprot.org/uniprot/{})'.format(proteinID, proteinID)
    hpa_link = '[Protein {}:](https://www.proteinatlas.org/search/{})'.format(geneName, geneName)

    df_summary_new = pd.DataFrame(data_summary.loc[input_value]).reset_index()
    df_summary_new.columns = ['entry', 'value']
    summary_single_protein = df_summary_new.to_dict('records')

    if type(data_peptide.loc[input_value])==pd.Series:
        summary_peptides = [data_peptide.loc[input_value].to_dict()]
    else:
        summary_peptides = data_peptide.loc[input_value].to_dict('records')

    return figure, figure_cpno, data, table_header,uniprot_link,hpa_link,summary_single_protein, summary_peptides

################################################################################
################################################################################

@app.callback(
    [Output('AbundanceRankPlot_HEP', 'figure'),
     Output('AbundanceRankPlot_LSEC', 'figure'),
     Output('AbundanceRankPlot_KC', 'figure'),
     Output('AbundanceRankPlot_HSC', 'figure')],
    [Input('ProteinID_abundance_rank', 'value')]
)
def update_abundance_rank_plot(input_value):
    #celltype_header = 'hHEP'
    #df_new = prepare_data_for_rank(celltype_header)
    #df_filtered = df_new[df_new['Genename_ProteinID'] == input_value]
    df_hep=df_rank_hep.copy()
    df_hep['input value']=np.where(df_hep['Genename_ProteinID']==input_value, input_value, 'All proteins')
    figure_hep = px.scatter(df_hep, x='Abundance rank', y='hHEP', color='input value',
                        color_discrete_map={input_value:'crimson', 'All proteins':'#7db0de'},
                       labels={'hHEP':'LFQ intensity [Log10]'})
    figure_hep.update_traces(marker=dict(size=10,line=dict(width=0.5,color='#12395d')),selector=dict(mode='markers'))
    figure_hep.update_layout(title='Hepatocytes',
                        margin=dict(l=50, r=2, t=120, b=50),
                        legend={'x': 0.1, 'y': 1.2})

    df_lsec=df_rank_lsec.copy()
    df_lsec['input value']=np.where(df_lsec['Genename_ProteinID']==input_value, input_value, 'All proteins')
    figure_lsec = px.scatter(df_lsec, x='Abundance rank', y='hLSEC', color='input value',
                        color_discrete_map={input_value:'crimson', 'All proteins':'#7db0de'},
                       labels={'hLSEC':'LFQ intensity [Log10]'})
    figure_lsec.update_traces(marker=dict(size=10,line=dict(width=0.5,color='#12395d')),selector=dict(mode='markers'))
    figure_lsec.update_layout(title='Liver sinusoidal endothelial cell',
                        margin=dict(l=50, r=2, t=120, b=50),
                        legend={'x': 0.1, 'y': 1.2})

    df_kc=df_rank_kc.copy()
    df_kc['input value']=np.where(df_kc['Genename_ProteinID']==input_value, input_value, 'All proteins')
    figure_kc = px.scatter(df_kc, x='Abundance rank', y='hKC', color='input value',
                        color_discrete_map={input_value:'crimson', 'All proteins':'#7db0de'},
                       labels={'hKC':'LFQ intensity [Log10]'})
    figure_kc.update_traces(marker=dict(size=10,line=dict(width=0.5,color='#12395d')),selector=dict(mode='markers'))
    figure_kc.update_layout(title='Kupffer cell',
                        margin=dict(l=50, r=2, t=120, b=50),
                        legend={'x': 0.1, 'y': 1.2})

    df_hsc=df_rank_hsc.copy()
    df_hsc['input value']=np.where(df_hsc['Genename_ProteinID']==input_value, input_value, 'All proteins')
    figure_hsc = px.scatter(df_hsc, x='Abundance rank', y='hHSC', color='input value',
                        color_discrete_map={input_value:'crimson', 'All proteins':'#7db0de'},
                       labels={'hHSC':'LFQ intensity [Log10]'})
    figure_hsc.update_traces(marker=dict(size=10,line=dict(width=0.5,color='#12395d')),selector=dict(mode='markers'))
    figure_hsc.update_layout(title='Hepatic stellate cell',
                        margin=dict(l=50, r=2, t=120, b=50),
                        legend={'x': 0.1, 'y': 1.2})
    return figure_hep, figure_lsec, figure_kc, figure_hsc

################################################################################
################################################################################

@app.callback(
    Output('cell_type_marker_table', 'data'),
    [Input('cell_type_marker_input', 'value')]
)
def update_celltypemarker_table(input_value):
    df = df_unique_tb.copy()
    filtered_df = df[df[input_value].min(1) > 5]
    filtered_df = filtered_df.round(2)
    return(filtered_df.to_dict('records'))
################################################################################
################################################################################

@app.callback(
    Output('volcano_table', 'data'),
    [Input('ProteinID', 'value')]
)
def update_volcano_protein(input_value):
    df = dataset.copy()
    filtered_df = df[df['Leading Gene name'] == input_value]
    filtered_df = filtered_df.round(2)
    return(filtered_df.to_dict('records'))

################################################################################
################################################################################
@app.callback(
    Output('volcano_table_pathway', 'data'),
    [Input('pathway_inference', 'value')]
)
def update_volcano_table2(input_gobp):
    df = dataset.copy()
    proteinlist_gobp = kegg_dict[input_gobp]
    filtered_df_gobp = df[df['Leading Gene name'].isin(proteinlist_gobp)]
    filtered_df_gobp = filtered_df_gobp.round(2)
    return(filtered_df_gobp.to_dict('records'))

################################################################################
################################################################################

#call back to update volcano plot of plasma data in cirrhosis
@app.callback(
    dash.dependencies.Output('VolcanoPlot_plasma_cirrhosis', 'figure'),
    [dash.dependencies.Input('ProteinID', 'value'),
     dash.dependencies.Input('pathway_inference', 'value'),
     dash.dependencies.Input('fold-change-range-slider-plasma', 'value')]
)

def update_output_div(input_protein, input_gobp, input_fold_change):
    #return 'You\'ve entered "{}"'.format(input_value)
    df = df_plasma.copy()
    df_sig = df_plasma[df_plasma['Significant'] == '+']
    filtered_df = df[df['Leading Gene name'] == input_protein]
    proteinlist_gobp = kegg_dict[input_gobp]
    filtered_df_gobp = df[df['Leading Gene name'].isin(proteinlist_gobp)]

    fc_min, fc_max = input_fold_change[0], input_fold_change[1]
    sig_fc = (df_sig['Log2 fold change']<fc_min)|(df_sig['Log2 fold change']>fc_max)
    x1 = df_sig[sig_fc]['Log2 fold change']
    y1 = df_sig[sig_fc]['-Log p-value']

    traces = [{'x': df['Log2 fold change'], 'y': df['-Log p-value'],
    'mode': 'markers',
    'marker':{'symbol':0,
                'size': 8,
                'color': '#EDEBEC',
                'line': dict(color='black',width=0.3)},
    'opacity':0.7,
    'name':'all proteins', 'text': df['Leading Gene name']},
    {'x': x1, 'y': y1,
    'mode': 'markers',
    'marker': {'symbol':0,
                'opacity': 0.8,
                'size': 8,
                'color': '#C70039',
                'line': dict(color='white',width=0.3)},
    'name':'significant proteins'}]
    traces.append(dict(x = list(filtered_df['Log2 fold change']),
                    y = list(filtered_df['-Log p-value']),
                    text = filtered_df['Leading Gene name'],
                    mode = 'markers',
                    marker = {'symbol':0,
                            'size': 8,
                            'color': 'yellow',
                            'line': dict(color='black',width=0.3)},
                    name = input_protein))

    traces.append(dict(x = list(filtered_df_gobp['Log2 fold change']),
                    y = list(filtered_df_gobp['-Log p-value']),
                    mode = 'markers',
                    marker = {'size': 8, 'symbol': 0, 'color' : '#243E58',
                            'line': dict(color='white',width=0.3)},
                    name = input_gobp))
    return {'data': traces,
            'layout': {'title':'Plasma',
                        'legend': {'x': 0, 'y': 1.15},
                        'xaxis':{'title':'Log2 FC Cirrhosis/Control'},
                        'yaxis':{'title':'-Log10 p-value'}}}

################################################################################
################################################################################

#call back to update volcano plot of liver data in cirrhosis
@app.callback(
    dash.dependencies.Output('VolcanoPlot', 'figure'),
    [dash.dependencies.Input('ProteinID', 'value'),
     dash.dependencies.Input('pathway_inference', 'value'),
     dash.dependencies.Input('fold-change-range-slider', 'value')]
)

def update_output_div(input_protein, input_gobp, input_fold_change):
    #return 'You\'ve entered "{}"'.format(input_value)

    filtered_df = df[df['Leading Gene name'] == input_protein]
    proteinlist_gobp = kegg_dict[input_gobp]
    filtered_df_gobp = df[df['Leading Gene name'].isin(proteinlist_gobp)]

    fc_min, fc_max = input_fold_change[0], input_fold_change[1]
    sig_fc = (df_sig['Log2 fold change']<fc_min)|(df_sig['Log2 fold change']>fc_max)
    x1 = df_sig[sig_fc]['Log2 fold change']
    y1 = df_sig[sig_fc]['-Log p-value']

    traces = [{'x': df['Log2 fold change'], 'y': df['-Log p-value'],
    'mode': 'markers',
    'marker':{'symbol':0,
                'size': '8',
                'color': '#EDEBEC',
                'line': dict(color='black',width=0.3)},
    'opacity': 0.8,
    'name':'all proteins', 'text': df['Leading Gene name']},
    {'x': x1, 'y': y1,
    'mode': 'markers',
    'marker': {'symbol':0,
                'opacity': 0.8,
                'size': '8',
                'color': '#C70039',
                'line': dict(color='white',width=0.3)},
    'name':'significant proteins'}]
    traces.append(dict(x = list(filtered_df['Log2 fold change']),
                    y = list(filtered_df['-Log p-value']),
                    text = filtered_df['Leading Gene name'],
                    mode = 'markers',
                    marker = {'symbol':0,
                                'size': '8',
                                'color': 'yellow',
                                'line': dict(color='black',width=0.3)},
                    name = input_protein))

    traces.append(dict(x = list(filtered_df_gobp['Log2 fold change']),
                    y = list(filtered_df_gobp['-Log p-value']),
                    mode = 'markers',
                    marker = {'symbol':0,
                            'size': '8',
                            'color': '#243E58',
                            'line': dict(color='white',width=0.3)},
                    opacity = 1,
                    name = input_gobp))
    return {'data': traces,
            'layout': {'title':'Liver',
                        'legend': {'x': 0, 'y': 1.15},
                        'xaxis':{'title':'Log2 FC Cirrhosis/Control'},
                        'yaxis':{'title':'-Log10 p-value'}}}

################################################################################
################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
