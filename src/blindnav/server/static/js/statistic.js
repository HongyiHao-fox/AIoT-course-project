import echarts from 'echarts'

const chart = echarts.init(document.getElementById('main'));

const option = {
    title: {
        text: 'ECharts Example'
    },
    tooltip: {},
    xAxis: {
        data: ['A', 'B', 'C', 'D', 'E', 'F']
    },
    yAxis: {},
    series: [{
        name: 'Example',
        type: 'bar',
        data: [5, 20, 36, 10, 10, 20]
    }]
};

chart.setOption(option);