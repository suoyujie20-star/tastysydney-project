import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import plotly.io as pio 

GEOJSON_PATH = 'Suburb_EPSG4326.json'
RESTAURANTS_CSV = 'restaurants_5000.csv'
STATS_CSV = 'sydney_restaurant_stats_2015_2025.csv'
# Load csv data 
df = pd.read_csv(RESTAURANTS_CSV, dtype={'restaurant_id': str})
# normalize
if 'suburb' not in df.columns:
    raise ValueError("restaurants CSV must contain a 'suburb' column")
df['suburb_norm'] = df['suburb'].astype(str).str.strip()

stats_df = pd.read_csv(STATS_CSV)
stats_df.columns = [c.strip() for c in stats_df.columns]

# Load GeoJSON  
with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
    gj = json.load(f)
def find_features(obj):
    if isinstance(obj, dict):
        if 'features' in obj and isinstance(obj['features'], list):
            return obj['features']
        for v in obj.values():
            res = find_features(v)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = find_features(item)
            if res is not None:
                return res
    return None
features = find_features(gj)
if features is None:
    raise ValueError("GeoJSON 'features' not found or not a supported structure.")

gdf = gpd.GeoDataFrame.from_features(features)
# detect suburb name
suburb_col = next((c for c in gdf.columns if 'suburb' in c.lower() or 'name' in c.lower()), None)
if suburb_col is None:
    sample_props = features[0].get('properties', {}) if features else {}
    suburb_col = next(iter(sample_props.keys()), None)
gdf['suburb_name_norm'] = gdf[suburb_col].astype(str).str.strip() if suburb_col else gdf.index.astype(str)

# add average consumption to geo features
cons = df.groupby('suburb_norm')['price_range_AUD'].mean().reset_index().rename(columns={'price_range_AUD':'avg_price_AUD'})
mean_price = df['price_range_AUD'].mean()
gdf = gdf.merge(cons, left_on='suburb_name_norm', right_on='suburb_norm', how='left')
gdf['avg_price_AUD'] = gdf['avg_price_AUD'].fillna(mean_price)

for feat in features:
    props = feat.setdefault('properties', {})
    name = str(props.get(suburb_col, '')).strip() if suburb_col else ''
    vals = gdf.loc[gdf['suburb_name_norm'] == name, 'avg_price_AUD']
    props['avg_price_AUD'] = float(vals.values[0]) if not vals.empty else float(mean_price)

geojson_fixed = {'type': 'FeatureCollection', 'features': features}
#  Data processing
if 'country' not in df.columns:
    df['country'] = df.get('category', 'Other').fillna('Other')
if 'dine_way' not in df.columns:
    np.random.seed(0)
    df['dine_way'] = np.random.choice(['Dine-in','Pick-up','Delivery'], size=len(df), p=[0.6,0.2,0.2])

# price range 
def price_band(p):
    if p <= 30: return '0-30'
    if p <= 60: return '30-60'
    if p <= 90: return '60-90'
    if p <= 120: return '90-120'
    return '120+'
df['price_band'] = df['price_range_AUD'].apply(price_band)
price_colors = {'0-30':'#2ecc71','30-60':'#3498db','60-90':'#f1c40f','90-120':'#e67e22','120+':'#e74c3c'}
df['color'] = df['price_band'].map(price_colors)

styles_all = sorted(df['style'].dropna().unique()) if 'style' in df.columns else []
move_to_style = ['Bakery','Bar','Cafe','Fast Food']
styles = sorted(set(styles_all).union(move_to_style))
# exclude styles out of countries
countries = sorted([c for c in df['country'].unique() if c not in move_to_style])
dineways = ['Dine-in','Pick-up','Delivery']
# selsct theme 
pio.templates["tastysydney"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", color="#1f2d3d"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(x=0.02, font=dict(size=18, color="#2c7be5"))
    )
)
px.defaults.template = "tastysydney"
#  setup Dash and style 
app = Dash(__name__)
server = app.server

PRIMARY = '#2c7be5'
BG = '#f3f7fc'
CARD_BG = 'linear-gradient(145deg,#ffffff 0%, #f7fbff 100%)'
SHADOW = '0 10px 30px rgba(11,22,60,0.08)'

card_style = {'background': CARD_BG, 'borderRadius': '12px', 'padding': '14px', 'boxShadow': SHADOW}
small_card = card_style.copy(); small_card.update({'padding':'10px'})
# Layout components (A, B, C, D, E)
# Header
header_bar = html.Div([
    html.Div("🍽️ TASTYSYDNEY", style={'fontWeight':'900','fontSize':'28px','color':'#fff','flex':'1'}),
    html.Div("Multicultural Dining Landscapes of Sydney", style={'textAlign':'center','flex':'2','color':'rgba(255,255,255,0.95)','fontWeight':'600'}),
    html.Div("", style={'flex':'1'})
], style={'display':'flex','alignItems':'center','justifyContent':'space-between','padding':'14px 28px','background':'linear-gradient(90deg,#2c7be5,#7f7bff)','boxShadow':'0 6px 20px rgba(0,0,0,0.12)','borderBottom':'3px solid rgba(0,0,0,0.06)'})
# Filters (A)
filters_panel = html.Div([
    html.H3("🎛️ Choose your preference", style={'color':PRIMARY,'fontWeight':'700'}),
    dcc.Input(id='search-input', type='text', placeholder='🔎 Search a restaurant (press Enter)...', debounce=True,
              style={'width':'100%','padding':'8px 10px','borderRadius':'8px','border':'1px solid #e6eef6','marginBottom':'12px'}),
    html.Div([
        html.Label("Country", style={'fontWeight':'600'}), 
        dcc.Checklist(id='country-check', options=[{'label':c,'value':c} for c in countries], value=countries[:6],
                      style={'height':'110px','overflowY':'auto','padding':'6px','background':'#fff','borderRadius':'8px','border':'1px solid #eef3fb'})
    ], style={'marginBottom':'10px'}),
    html.Div([
        html.Div([html.Label("Style", style={'fontWeight':'600'}),
                  dcc.Checklist(id='style-check', options=[{'label':s,'value':s} for s in styles], value=styles[:6],
                                style={'height':'110px','overflowY':'auto','padding':'6px','background':'#fff','borderRadius':'8px','border':'1px solid #eef3fb'})], style={'width':'48%','display':'inline-block'}),
        html.Div([html.Label("Dine Way", style={'fontWeight':'600'}),
                  dcc.Checklist(id='dineway-check', options=[{'label':d,'value':d} for d in dineways], value=dineways,
                                style={'height':'110px','overflowY':'auto','padding':'6px','background':'#fff','borderRadius':'8px','border':'1px solid #eef3fb'})], style={'width':'48%','display':'inline-block','marginLeft':'4%'})
    ], style={'marginBottom':'12px'}),
    html.Label("Price Range (AUD)", style={'fontWeight':'600'}),
    dcc.RangeSlider(id='price-slider', min=0, max=120, step=5, value=[0,120], marks={0:'0',30:'30',60:'60',90:'90',120:'120+'}),
    html.Br(),
    html.Label("Rating (Google)", style={'fontWeight':'600'}),
    dcc.RangeSlider(id='rating-slider', min=1, max=5, step=0.1, value=[1,5], marks={1:'1',2:'2',3:'3',4:'4',5:'5'}),
    html.Br(),
    html.Label("Suburb", style={'fontWeight':'600'}),
    dcc.Dropdown(id='suburb-drop', options=[{'label':s,'value':s} for s in sorted(df['suburb_norm'].unique())], multi=True, placeholder='Select suburb(s)'),
    html.Br(),
    html.Label("Platform", style={'fontWeight':'600'}),
    dcc.Dropdown(id='plat-drop', options=[{'label':p,'value':p} for p in sorted({x for row in df['platforms'].dropna().str.split(',') for x in row})], multi=True, placeholder='Select platform(s)'),
    html.Br(),
    html.Button("Reset Filters", id='reset-btn', style={'background':PRIMARY,'color':'#fff','border':'none','padding':'8px 12px','borderRadius':'8px','cursor':'pointer'})
], style={**card_style, 'width':'25%','height':'92vh','overflowY':'auto'})
# Map (B)
map_panel = html.Div([
    html.Div([html.H3("🗺️ Sydney Restaurant Overview", style={'color':PRIMARY,'fontWeight':'700'})]),
    html.Div([
        dcc.Input(id='map-search', type='text', placeholder='Search restaurant and press Enter', debounce=True, style={'width':'65%','padding':'6px','borderRadius':'6px','border':'1px solid #e6eef6','marginRight':'8px'}),
        html.Button("Search", id='map-search-btn', n_clicks=0, style={'background':'#27ae60','color':'white','border':'none','padding':'6px 10px','borderRadius':'6px','cursor':'pointer'})
    ], style={'display':'flex','alignItems':'center','marginBottom':'8px'}),
    html.Div([html.Label("Map Mode:"), dcc.RadioItems(id='map-mode', options=[{'label':'OpenStreetMap','value':'osm'},{'label':'Choropleth (clean)','value':'choropleth'}], value='osm', inline=True)], style={'marginBottom':'8px'}),
    dcc.Graph(id='map-fig', style={'height':'78vh'}),
    dcc.Store(id='filtered-store'),
    dcc.Store(id='rank-offset-store', data=0)  # store offset for load more
], style={**card_style, 'width':'50%'})
# Area (C)
area_panel = html.Div([
    html.H3("📊 Range Statistics", style={'color':PRIMARY,'fontWeight':'700'}),
    html.Div(id='area-content', style={'minHeight':'320px'})
], style={**card_style, 'width':'25%'})

# Rank table (D)
rank_section = html.Div([
    html.H3("🏆 Rank Table", style={'color':PRIMARY,'fontWeight':'700'}),
    html.Div([
        html.Div([html.Label("Filter Country"), dcc.Dropdown(id='table-country-filter', options=[{'label':c,'value':c} for c in countries], multi=True)], style={'width':'48%','display':'inline-block'}),
        html.Div([html.Label("Filter Style"), dcc.Dropdown(id='table-style-filter', options=[{'label':s,'value':s} for s in styles], multi=True)], style={'width':'48%','display':'inline-block','marginLeft':'4%'})
    ], style={'marginBottom':'10px'}),
    html.Table([
        html.Thead(html.Tr([html.Th("Name"), html.Th("Country"), html.Th("Style"),
                            html.Th(html.Button("Price", id='sort-price', n_clicks=0, style={'background':'none','border':'none','color':PRIMARY,'cursor':'pointer'})),
                            html.Th(html.Button("Google Rating", id='sort-google', n_clicks=0, style={'background':'none','border':'none','color':PRIMARY,'cursor':'pointer'})),
                            html.Th(html.Button("Uber Rating", id='sort-uber', n_clicks=0, style={'background':'none','border':'none','color':PRIMARY,'cursor':'pointer'}))
                           ])),
        html.Tbody(id='rank-table-body')
    ], style={'width':'100%','borderCollapse':'collapse'}),
    html.Div([html.Button("Load more", id='load-more', n_clicks=0, style={'background':'#6c757d','color':'#fff','border':'none','padding':'8px 12px','borderRadius':'8px','cursor':'pointer'})], style={'marginTop':'10px','textAlign':'center'})
], style={**card_style, 'marginTop':'12px'})

# Trends (E)
trend_section = html.Div([
    html.H3("📈 Trends (2015 - 2025)", style={'color':PRIMARY,'fontWeight':'700'}),
    dcc.Graph(id='trend-counts', style={'height':'300px'}),
    dcc.Graph(id='trend-consumption', style={'height':'300px'}),
    dcc.Graph(id='trend-rating', style={'height':'300px'})
], style={**card_style, 'marginTop':'12px'})

# App layout
app.layout = html.Div([header_bar, html.Div([filters_panel, map_panel, area_panel], style={'display':'flex','gap':'14px','padding':'14px'}), rank_section, trend_section], style={'background':BG, 'minHeight':'100vh', 'paddingBottom':'30px'})
#  Callbacks 
# Reset filters
@app.callback(
    Output('country-check','value'),
    Output('style-check','value'),
    Output('dineway-check','value'),
    Output('price-slider','value'),
    Output('rating-slider','value'),
    Output('suburb-drop','value'),
    Output('plat-drop','value'),
    Input('reset-btn','n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n):
    return countries[:6], styles[:6], dineways, [0,120], [1,5], [], []
# Map update (main callback) -> returns figure and filtered-store
@app.callback(
    Output('map-fig','figure'),
    Output('filtered-store','data'),
    Input('country-check','value'),
    Input('style-check','value'),
    Input('dineway-check','value'),
    Input('price-slider','value'),
    Input('rating-slider','value'),
    Input('suburb-drop','value'),
    Input('plat-drop','value'),
    Input('map-mode','value'),
    Input('map-search-btn','n_clicks'),
    State('map-search','value'),
    Input('search-input','value'),
    prevent_initial_call=False
)
def update_map(countries_sel, styles_sel, dineway_sel, price_range, rating_range, suburb_sel, plat_sel, map_mode, search_btn_clicks, map_search_value, search_input_value):
    dff = df.copy()
    if countries_sel:
        dff = dff.loc[dff['country'].isin(countries_sel)].reset_index(drop=True)
    if styles_sel:
        dff = dff.loc[dff['style'].isin(styles_sel)].reset_index(drop=True)
    if dineway_sel:
        dff = dff.loc[dff['dine_way'].isin(dineway_sel)].reset_index(drop=True)
    dff = dff.loc[(dff['price_range_AUD']>=price_range[0]) & (dff['price_range_AUD']<=price_range[1])].reset_index(drop=True)
    dff = dff.loc[(dff['google_rating']>=rating_range[0]) & (dff['google_rating']<=rating_range[1])].reset_index(drop=True)
    if suburb_sel:
        dff = dff.loc[dff['suburb_norm'].isin(suburb_sel)].reset_index(drop=True)
    if plat_sel:
        dff = dff.loc[dff['platforms'].apply(lambda s: any(p in s for p in plat_sel) if isinstance(s,str) else False)].reset_index(drop=True)
        
    cap = min(len(dff), 3000)
    dff = dff.sort_values(['google_rating','uber_rating'], ascending=False).head(cap).reset_index(drop=True)

    search_value = None
    if map_search_value and isinstance(map_search_value, str) and map_search_value.strip():
        search_value = map_search_value.strip()
    elif search_input_value and isinstance(search_input_value, str) and search_input_value.strip():
        search_value = search_input_value.strip()

    suburb_prices = np.array([feat['properties'].get('avg_price_AUD', mean_price) for feat in features], dtype=float)
    qbins = np.quantile(suburb_prices, np.linspace(0,1,11))  # 10-quantiles

    # build figure
    if map_mode == 'choropleth':
        choropleth = go.Choroplethmapbox(
            geojson=geojson_fixed,
            locations=[feat['properties'].get(suburb_col,'') for feat in features],
            z=[feat['properties'].get('avg_price_AUD', mean_price) for feat in features],
            colorscale='YlOrRd',
            zmin=qbins[0],
            zmax=qbins[-1],
            marker_opacity=0.85,
            marker_line_width=0.6,
            featureidkey=f"properties.{suburb_col}",
            colorbar=dict(title="Avg Consumption (AUD)", orientation='h', x=0.5, xanchor='center', y=-0.12, len=0.6)
        )
        fig = go.Figure(choropleth)
        fig.update_layout(mapbox_style="carto-positron")
    else:
        underlay = go.Choroplethmapbox(
            geojson=geojson_fixed,
            locations=[feat['properties'].get(suburb_col,'') for feat in features],
            z=[feat['properties'].get('avg_price_AUD', mean_price) for feat in features],
            colorscale='Blues',
            zmin=qbins[0],
            zmax=qbins[-1],
            marker_opacity=0.06,
            marker_line_width=0.3,
            featureidkey=f"properties.{suburb_col}",
            showscale=False
        )
        fig = go.Figure(underlay)
        fig.update_layout(mapbox_style="open-street-map")

    # add restaurant points
    fig.add_trace(go.Scattermapbox(
        lat=dff['latitude'], lon=dff['longitude'],
        mode='markers',
        marker=dict(size=7, color=[price_colors.get(pb,'#888') for pb in dff['price_band']], opacity=0.95),
        text=dff['name'],
        hovertext=[f"{r['name']}<br>💰{r['price_range_AUD']} AUD | {r['price_band']}<br>⭐{r['google_rating']}" for _, r in dff.iterrows()],
        hoverinfo='text',
        name='restaurants'
    ))
    if search_value:
        matched = dff.loc[dff['name'].str.contains(search_value, case=False, na=False)]
        if not matched.empty:
            fig.add_trace(go.Scattermapbox(
                lat=matched['latitude'], lon=matched['longitude'],
                mode='markers+text',
                marker=dict(size=14, color='red', symbol='star'),
                text=matched['name'],
                textposition='top right',
                hoverinfo='text',
                name='highlight'
            ))
            try:
                lat0 = float(matched.iloc[0]['latitude']); lon0 = float(matched.iloc[0]['longitude'])
                fig.update_layout(mapbox_center={"lat": lat0, "lon": lon0}, mapbox_zoom=13)
            except Exception:
                pass

    fig.update_layout(mapbox_zoom=10.7, mapbox_center={"lat": -33.8688, "lon": 151.2093}, margin=dict(l=0,r=0,t=0,b=40), uirevision='map-state')

    return fig, dff.to_dict('records')
# Area (C) callback
@app.callback(
    Output('area-content','children'),
    Input('map-fig','clickData'),
    Input('suburb-drop','value'),
    State('filtered-store','data')
)
def area_content(clickData, suburb_sel, filtered):
    selected = None
    if clickData and 'points' in clickData:
        name = clickData['points'][0].get('text')
        if name:
            matched = df.loc[df['name'] == name, 'suburb_norm']
            if not matched.empty:
                selected = matched.values[0]
    if not selected and suburb_sel:
        selected = suburb_sel[0] if isinstance(suburb_sel, list) and suburb_sel else None
    if not selected and filtered:
        arr = filtered if isinstance(filtered, list) else []
        if arr:
            selected = arr[0].get('suburb_norm')

    if not selected:
        return html.Div("Select an area and explore more", style={'textAlign':'center','marginTop':'40%','fontSize':'16px','color':'#666'})

    subdf = df.loc[df['suburb_norm'] == selected].reset_index(drop=True)
    total = len(subdf)
    avg_rating = round(subdf['google_rating'].mean(),2) if total>0 else 0.0
    avg_price_local = round(subdf['price_range_AUD'].mean(),2) if total>0 else 0.0

    summary = html.Div([
        html.H4(selected, style={'marginBottom':'6px'}),
        html.P(f"Total restaurants: {total}", style={'margin':'4px 0'}),
        html.P(f"Average rating (Google): {avg_rating}", style={'margin':'4px 0'}),
        html.P(f"Average consumption (AUD): ${avg_price_local}", style={'margin':'4px 0'})
    ], style={'padding':'8px','background':'#fbfdff','borderRadius':'8px','marginBottom':'12px'})

    pie_df = subdf['country'].value_counts().reset_index(); pie_df.columns=['country','count']
    fig_pie = px.pie(pie_df, names='country', values='count', hole=0.25)
    fig_pie.update_traces(textinfo='percent+label', textposition='inside', textfont_size=12)
    fig_pie.update_layout(margin=dict(t=8,b=8,l=8,r=8), legend=dict(orientation='h', y=-0.15, x=0.05))

    return html.Div([summary, dcc.Graph(figure=fig_pie, config={'displayModeBar':False}, style={'height':'320px'})])
# Rank table (D) callback
@app.callback(
    Output('rank-table-body','children'),
    Output('rank-offset-store','data'),
    Input('table-country-filter','value'),
    Input('table-style-filter','value'),
    Input('sort-price','n_clicks'),
    Input('sort-google','n_clicks'),
    Input('sort-uber','n_clicks'),
    Input('load-more','n_clicks'),
    State('filtered-store','data'),
    State('rank-offset-store','data')
)
def update_rank_table(country_filter, style_filter, n_price, n_google, n_uber, load_more_clicks, store, offset):
    dff = pd.DataFrame(store) if store else df.copy()
    if country_filter:
        dff = dff.loc[dff['country'].isin(country_filter)].reset_index(drop=True)
    if style_filter:
        dff = dff.loc[dff['style'].isin(style_filter)].reset_index(drop=True)

    clicks = {'price': n_price or 0, 'google': n_google or 0, 'uber': n_uber or 0}
    if max(clicks.values()) == 0:
        sort_col, asc = ('google_rating', False)
    else:
        last = max(clicks, key=clicks.get)
        sort_col = {'price':'price_range_AUD','google':'google_rating','uber':'uber_rating'}[last]
        asc = (clicks[last] % 2 == 1)
    dff = dff.sort_values(sort_col, ascending=asc).reset_index(drop=True)
    if offset is None:
        offset = 0
    per_page = 10
  
    if load_more_clicks:
        offset = min(len(dff), offset + per_page)
    else:
        offset = min(len(dff), per_page) if offset == 0 else offset

    shown = dff.head(offset if offset>0 else per_page)

    rows = []
    for _, r in shown.iterrows():
        rows.append(html.Tr([
            html.Td(r.get('name',''), style={'padding':'8px','borderBottom':'1px solid #f1f4f8'}),
            html.Td(r.get('country',''), style={'padding':'8px','borderBottom':'1px solid #f1f4f8'}),
            html.Td(r.get('style',''), style={'padding':'8px','borderBottom':'1px solid #f1f4f8'}),
            html.Td(f"${r.get('price_range_AUD','')}", style={'padding':'8px','borderBottom':'1px solid #f1f4f8'}),
            html.Td(r.get('google_rating',''), style={'padding':'8px','borderBottom':'1px solid #f1f4f8'}),
            html.Td(r.get('uber_rating',''), style={'padding':'8px','borderBottom':'1px solid #f1f4f8'})
        ]))
    return rows, offset
# Trends (E) callback
@app.callback(
    Output('trend-counts','figure'),
    Output('trend-consumption','figure'),
    Output('trend-rating','figure'),
    Input('suburb-drop','value')
)
def update_trends(suburb_sel):
    years = sorted(stats_df['year'].unique())
    selected_suburbs = suburb_sel if suburb_sel else []
    if not selected_suburbs:
        top_areas = stats_df.groupby('suburb')['total_count'].sum().sort_values(ascending=False).head(4).index.tolist()
    else:
        top_areas = selected_suburbs

    agg = stats_df.groupby(['year','suburb'], as_index=False).agg({'total_count':'sum','avg_price_AUD':'mean','avg_rating':'mean'})

    # Chart 1: counts
    fig_counts = go.Figure()
    for area in top_areas:
        tmp = agg[agg['suburb'] == area][['year','total_count']].set_index('year')
        tmp = tmp.reindex(years, fill_value=0).reset_index()
        fig_counts.add_trace(go.Bar(x=tmp['year'], y=tmp['total_count'], name=area))
    fig_counts.update_layout(barmode='group', title='Restaurant counts (2015–2025)', xaxis_title='Year', yaxis_title='Count', template='plotly_white', height=320)

    # Chart 2: avg consumption
    fig_cons = go.Figure()
    for area in top_areas:
        tmp = agg[agg['suburb'] == area][['year','avg_price_AUD']].set_index('year')
        tmp = tmp.reindex(years).interpolate().reset_index()
        fig_cons.add_trace(go.Scatter(x=tmp['year'], y=tmp['avg_price_AUD'], mode='lines+markers', name=area))
    fig_cons.update_layout(title='Average consumption (AUD) by area (2015–2025)', xaxis_title='Year', yaxis_title='Avg Consumption (AUD)', template='plotly_white', height=320)

    # Chart 3: avg rating
    fig_rating = go.Figure()
    for area in top_areas:
        tmp = agg[agg['suburb'] == area][['year','avg_rating']].set_index('year')
        tmp = tmp.reindex(years).interpolate().reset_index()
        fig_rating.add_trace(go.Scatter(x=tmp['year'], y=tmp['avg_rating'], mode='lines+markers', name=area))
    fig_rating.update_layout(title='Average Google rating by area (2015–2025)', xaxis_title='Year', yaxis_title='Avg Rating', template='plotly_white', height=320)

    return fig_counts, fig_cons, fig_rating
#  Run server 
server = app.server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)