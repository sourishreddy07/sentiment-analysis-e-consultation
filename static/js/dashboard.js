/**
 * Dashboard & Interactive Sentiment Analytics Client Script
 * Handles real-time single comment inference, KPI live updates,
 * and Chart.js dynamic visualizations.
 */

let distributionChartInstance = null;
let domainChartInstance = null;
let benchmarkChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Chart.js visualizations
    if (typeof initialStats !== 'undefined' && initialStats) {
        initDistributionChart(initialStats);
        initDomainChart(initialStats);
    }
    if (typeof initialMetrics !== 'undefined' && initialMetrics) {
        initBenchmarkChart(initialMetrics);
    }

    // 2. Setup Form Submission Listener
    const commentForm = document.getElementById('commentForm');
    if (commentForm) {
        commentForm.addEventListener('submit', handleCommentSubmission);
    }
});

// Tab Switcher
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(
        btn => btn.getAttribute('onclick').includes(tabName)
    );
    if (activeBtn) activeBtn.classList.add('active');

    const targetContent = document.getElementById(`tab-${tabName}`);
    if (targetContent) {
        targetContent.classList.add('active');
        // Trigger resize on charts to prevent rendering artifacts
        if (tabName === 'distribution' && distributionChartInstance) distributionChartInstance.resize();
        if (tabName === 'domains' && domainChartInstance) domainChartInstance.resize();
        if (tabName === 'benchmarks' && benchmarkChartInstance) benchmarkChartInstance.resize();
    }
}

// Handle Single Comment Submission
async function handleCommentSubmission(e) {
    e.preventDefault();

    const commentInput = document.getElementById('commentText');
    const domainInput = document.getElementById('domainSelect');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const spinner = analyzeBtn.querySelector('.spinner');

    const comment = commentInput.value.trim();
    const domain = domainInput.value;

    if (!comment) return;

    // Loading State
    analyzeBtn.disabled = true;
    btnText.textContent = 'Analyzing...';
    spinner.style.display = 'inline-block';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ comment, domain })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            alert('Analysis Error: ' + (data.error || 'Failed to process comment'));
            return;
        }

        // Render Prediction Result
        displayPredictionResult(data);

        // Update live KPIs & Table
        if (data.db_saved) {
            updateKPIs(data.sentiment);
            prependRecentComment(data);
            refreshCharts();
        } else if (data.db_warning) {
            console.warn('DB Notice:', data.db_warning);
        }

    } catch (err) {
        console.error('Submission error:', err);
        alert('Network or server error while connecting to Flask backend.');
    } finally {
        analyzeBtn.disabled = false;
        btnText.textContent = 'Analyze Sentiment';
        spinner.style.display = 'none';
    }
}

// Display Result Card
function displayPredictionResult(data) {
    const card = document.getElementById('predictionResult');
    const badge = document.getElementById('sentimentBadge');
    const confScore = document.getElementById('confidenceScore');
    const confPercent = document.getElementById('confidencePercent');
    const confProgress = document.getElementById('confidenceProgress');
    const preprocessed = document.getElementById('preprocessedText');
    const timestamp = document.getElementById('resultTimestamp');

    // VADER elements
    const vaderBadge = document.getElementById('vaderBadge');
    const vaderCompound = document.getElementById('vaderCompound');
    const vaderPos = document.getElementById('vaderPos');
    const vaderNeg = document.getElementById('vaderNeg');
    const vaderNeu = document.getElementById('vaderNeu');

    // Sentiment badge
    const sentiment = data.sentiment.toLowerCase();
    badge.className = `sentiment-badge ${sentiment}`;
    badge.textContent = sentiment.toUpperCase();

    // Confidence
    const conf = Math.round(data.confidence * 100);
    confScore.textContent = `${conf}% Confidence`;
    confPercent.textContent = `${conf}%`;
    confProgress.className = `progress-fill ${sentiment}`;
    confProgress.style.width = `${conf}%`;

    // Preprocessed text
    preprocessed.textContent = data.preprocessed_comment || '(empty)';
    timestamp.textContent = new Date().toLocaleTimeString();

    // VADER Baseline
    if (data.vader) {
        const vs = data.vader.sentiment.toLowerCase();
        vaderBadge.className = `vader-badge ${vs}`;
        vaderBadge.textContent = vs.toUpperCase();
        vaderCompound.textContent = data.vader.compound;
        vaderPos.textContent = data.vader.positive;
        vaderNeg.textContent = data.vader.negative;
        vaderNeu.textContent = data.vader.neutral;
    }

    // Model Explainability (Extension #10)
    renderExplainabilitySection(data.explanation);

    card.style.display = 'block';
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Render Explainability Section
function renderExplainabilitySection(exp) {
    const section = document.getElementById('explainabilitySection');
    if (!section) return;

    if (!exp) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';

    const summaryText = document.getElementById('explainSummaryText');
    const mlBadge = document.getElementById('explainMlBadge');
    const vaderBadge = document.getElementById('explainVaderBadge');
    const agreeBadge = document.getElementById('explainAgreementBadge');
    const posChips = document.getElementById('posContribChips');
    const negChips = document.getElementById('negContribChips');
    const tbody = document.getElementById('explainFeaturesTbody');

    if (summaryText) summaryText.textContent = exp.summary || 'No explanatory features found.';

    if (mlBadge) {
        const mlSent = (exp.predicted_sentiment || 'neutral').toLowerCase();
        mlBadge.className = `sentiment-badge badge-sm ${mlSent}`;
        mlBadge.textContent = mlSent.toUpperCase();
    }

    if (vaderBadge) {
        const vSent = (exp.vader_sentiment || 'neutral').toLowerCase();
        vaderBadge.className = `sentiment-badge badge-sm ${vSent}`;
        vaderBadge.textContent = vSent.toUpperCase();
    }

    if (agreeBadge) {
        if (exp.agreement) {
            agreeBadge.className = 'agreement-badge agree-yes';
            agreeBadge.textContent = '✓ Agree';
        } else {
            agreeBadge.className = 'agreement-badge agree-no';
            agreeBadge.textContent = '≠ Diverge';
        }
    }

    // Positive chips
    if (posChips) {
        posChips.innerHTML = '';
        const pFeats = exp.positive_features || [];
        if (pFeats.length === 0) {
            posChips.innerHTML = '<span class="no-feats-note">None identified</span>';
        } else {
            pFeats.forEach(f => {
                const chip = document.createElement('span');
                chip.className = 'contrib-chip chip-pos';
                chip.innerHTML = `<strong>${escapeHtml(f.word)}</strong> <span class="chip-score">+${f.contribution.toFixed(2)}</span>`;
                posChips.appendChild(chip);
            });
        }
    }

    // Negative chips
    if (negChips) {
        negChips.innerHTML = '';
        const nFeats = exp.negative_features || [];
        if (nFeats.length === 0) {
            negChips.innerHTML = '<span class="no-feats-note">None identified</span>';
        } else {
            nFeats.forEach(f => {
                const chip = document.createElement('span');
                chip.className = 'contrib-chip chip-neg';
                chip.innerHTML = `<strong>${escapeHtml(f.word)}</strong> <span class="chip-score">${f.contribution.toFixed(2)}</span>`;
                negChips.appendChild(chip);
            });
        }
    }

    // Features table
    if (tbody) {
        tbody.innerHTML = '';
        const topFeats = exp.top_features || [];
        if (topFeats.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted" style="padding: 0.75rem;">No non-zero TF-IDF vocabulary features matched this comment.</td></tr>';
        } else {
            topFeats.forEach(f => {
                const tr = document.createElement('tr');
                const isPos = f.direction === 'positive';
                tr.innerHTML = `
                    <td class="font-semibold">${escapeHtml(f.word)}</td>
                    <td class="text-right font-mono">${f.tfidf.toFixed(4)}</td>
                    <td class="text-right font-mono">${f.weight > 0 ? '+' : ''}${f.weight.toFixed(4)}</td>
                    <td class="text-right font-mono ${isPos ? 'text-positive' : 'text-negative'} font-bold">
                        ${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(4)}
                    </td>
                    <td>
                        <span class="pill-pct ${isPos ? 'pill-positive' : 'pill-negative'}">
                            ${isPos ? 'Positive' : 'Negative'}
                        </span>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Update KPI DOM counts
function updateKPIs(sentiment) {
    const totalEl = document.getElementById('kpi-total');
    const targetEl = document.getElementById(`kpi-${sentiment}`);
    const sessionCountEl = document.getElementById('sessionCommentsCount');

    if (totalEl) {
        const curTotal = parseInt(totalEl.textContent) || 0;
        totalEl.textContent = curTotal + 1;
    }
    if (sessionCountEl) {
        const curSess = parseInt(sessionCountEl.textContent) || 0;
        sessionCountEl.textContent = curSess + 1;
    }
    if (targetEl) {
        const curTarget = parseInt(targetEl.textContent) || 0;
        targetEl.textContent = curTarget + 1;
    }
}

// Prepend to Recent Comments Table
function prependRecentComment(data) {
    const tbody = document.getElementById('recentCommentsBody');
    if (!tbody) return;

    // Remove empty placeholder row if exists
    const emptyRow = tbody.querySelector('.empty-state');
    if (emptyRow) emptyRow.closest('tr').remove();

    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td>#${data.id || 'new'}</td>
        <td class="comment-cell" title="${data.comment}">${data.comment}</td>
        <td><span class="domain-tag">${data.domain}</span></td>
        <td><span class="badge badge-${data.sentiment}">${data.sentiment.charAt(0).toUpperCase() + data.sentiment.slice(1)}</span></td>
        <td>${Math.round(data.confidence * 100)}%</td>
        <td><span class="vader-chip">${data.vader ? data.vader.sentiment : '-'} (${data.vader ? data.vader.compound : '0'})</span></td>
        <td class="time-cell">Just now</td>
    `;
    tbody.insertBefore(tr, tbody.firstChild);
}

// Chart 1: Sentiment Distribution Doughnut
function initDistributionChart(stats) {
    const ctx = document.getElementById('distributionChart');
    if (!ctx) return;

    const total = stats.total || 0;
    const hasData = total > 0;
    const chartData = hasData
        ? [stats.positive || 0, stats.negative || 0, stats.neutral || 0]
        : [0, 0, 0];

    const data = {
        labels: ['Positive', 'Negative', 'Neutral / Borderline'],
        datasets: [{
            data: chartData,
            backgroundColor: ['#10b981', '#ef4444', '#6366f1'],
            borderWidth: 2,
            borderColor: '#ffffff',
            hoverOffset: 4
        }]
    };

    distributionChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { font: { family: 'Inter', size: 12 } }
                },
                title: {
                    display: true,
                    text: hasData ? 'Session Sentiment Breakdown' : 'Session Sentiment Breakdown (Awaiting Comments)',
                    font: { family: 'Inter', size: 14, weight: '600' }
                }
            },
            cutout: '65%'
        }
    });
}

// Chart 2: Consultation Domain Analysis (Grouped/Stacked Bar + Metrics Breakdown)
function renderDomainAnalytics(domainList) {
    const emptyState = document.getElementById('domainEmptyState');
    const chartContainer = document.getElementById('domainChartContainer');
    const metricsContainer = document.getElementById('domainMetricsContainer');
    const canvas = document.getElementById('domainChart');

    if (!canvas) return;

    // Normalization: handles either structured [{domain, positive, negative, neutral}] or legacy rows
    let normalized = [];
    if (Array.isArray(domainList) && domainList.length > 0) {
        if (domainList[0].positive !== undefined) {
            normalized = domainList;
        } else {
            const map = {};
            domainList.forEach(r => {
                const dom = r.domain || 'general';
                if (!map[dom]) map[dom] = { domain: dom, positive: 0, negative: 0, neutral: 0 };
                const s = (r.sentiment || '').toLowerCase();
                if (s in map[dom]) map[dom][s] += (r.count || 0);
            });
            normalized = Object.values(map);
        }
    }

    const hasData = normalized.length > 0;

    if (!hasData) {
        if (emptyState) emptyState.style.display = 'flex';
        if (chartContainer) chartContainer.style.display = 'none';
        if (metricsContainer) metricsContainer.innerHTML = '';
        if (domainChartInstance) {
            domainChartInstance.destroy();
            domainChartInstance = null;
        }
        return;
    }

    if (emptyState) emptyState.style.display = 'none';
    if (chartContainer) chartContainer.style.display = 'block';

    const labels = normalized.map(d => (d.domain || 'general').replace(/_/g, ' ').toUpperCase());
    const posData = normalized.map(d => d.positive || 0);
    const negData = normalized.map(d => d.negative || 0);
    const neuData = normalized.map(d => d.neutral || 0);

    if (domainChartInstance) {
        domainChartInstance.data.labels = labels;
        domainChartInstance.data.datasets[0].data = posData;
        domainChartInstance.data.datasets[1].data = negData;
        domainChartInstance.data.datasets[2].data = neuData;
        domainChartInstance.update();
    } else {
        domainChartInstance = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Positive',
                        data: posData,
                        backgroundColor: '#10b981',
                        borderRadius: 4
                    },
                    {
                        label: 'Negative',
                        data: negData,
                        backgroundColor: '#ef4444',
                        borderRadius: 4
                    },
                    {
                        label: 'Neutral',
                        data: neuData,
                        backgroundColor: '#6366f1',
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true,
                        ticks: { font: { family: 'Inter', size: 11, weight: '500' } },
                        grid: { display: false }
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        ticks: { precision: 0, stepSize: 1, font: { family: 'Inter', size: 11 } }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { font: { family: 'Inter', size: 12 }, boxWidth: 14 }
                    },
                    title: {
                        display: true,
                        text: 'Session Sentiment by Consultation Domain',
                        font: { family: 'Inter', size: 13, weight: '600' }
                    },
                    tooltip: {
                        callbacks: {
                            afterTitle: (items) => {
                                const idx = items[0].dataIndex;
                                const item = normalized[idx];
                                const total = (item.positive || 0) + (item.negative || 0) + (item.neutral || 0);
                                return `Total: ${total} comments`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Render domain breakdown metric cards
    if (metricsContainer) {
        metricsContainer.innerHTML = normalized.map(d => {
            const domainTitle = (d.domain || 'general')
                .replace(/_/g, ' ')
                .replace(/\b\w/g, c => c.toUpperCase());
            const total = (d.positive || 0) + (d.negative || 0) + (d.neutral || 0);
            return `
                <div class="domain-stat-card">
                    <div class="domain-card-head">
                        <span class="domain-card-name">${domainTitle}</span>
                        <span class="domain-card-total">${total} total</span>
                    </div>
                    <div class="domain-card-counts">
                        <span class="domain-pill pos"><span>Positive:</span> <strong>${d.positive || 0}</strong></span>
                        <span class="domain-pill neg"><span>Negative:</span> <strong>${d.negative || 0}</strong></span>
                        <span class="domain-pill neu"><span>Neutral:</span> <strong>${d.neutral || 0}</strong></span>
                    </div>
                </div>
            `;
        }).join('');
    }
}

function initDomainChart(stats) {
    renderDomainAnalytics(stats ? stats.domain_distribution : []);
}

// Chart 3: Model Benchmark Comparisons
function initBenchmarkChart(metrics) {
    const ctx = document.getElementById('benchmarkChart');
    if (!ctx || !metrics.models_comparison) return;

    const models = Object.keys(metrics.models_comparison);
    const accuracies = models.map(m => metrics.models_comparison[m].accuracy);
    const precisions = models.map(m => metrics.models_comparison[m].precision);
    const recalls = models.map(m => metrics.models_comparison[m].recall);
    const f1s = models.map(m => metrics.models_comparison[m].f1_score);

    benchmarkChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                { label: 'Accuracy %', data: accuracies, backgroundColor: '#0284c7' },
                { label: 'Precision %', data: precisions, backgroundColor: '#14b8a6' },
                { label: 'Recall %', data: recalls, backgroundColor: '#f59e0b' },
                { label: 'F1-Score %', data: f1s, backgroundColor: '#8b5cf6' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 80, max: 100, ticks: { callback: v => v + '%' } }
            },
            plugins: {
                legend: { position: 'bottom' },
                title: {
                    display: true,
                    text: 'Empirical Test Performance Comparison',
                    font: { family: 'Inter', size: 14, weight: '600' }
                }
            }
        }
    });
}

// Refresh Charts dynamically via /api/stats
async function refreshCharts() {
    try {
        const res = await fetch('/api/stats');
        const stats = await res.json();
        if (!stats) return;

        // Sync KPI numbers
        if (typeof stats.total !== 'undefined') {
            const totalEl = document.getElementById('kpi-total');
            const sessEl = document.getElementById('sessionCommentsCount');
            const posEl = document.getElementById('kpi-positive');
            const negEl = document.getElementById('kpi-negative');
            const neuEl = document.getElementById('kpi-neutral');
            const posPct = document.getElementById('kpi-positive-pct');
            const negPct = document.getElementById('kpi-negative-pct');

            if (totalEl) totalEl.textContent = stats.total;
            if (sessEl) sessEl.textContent = stats.total;
            if (posEl) posEl.textContent = stats.positive || 0;
            if (negEl) negEl.textContent = stats.negative || 0;
            if (neuEl) neuEl.textContent = stats.neutral || 0;
            if (posPct) posPct.textContent = `${stats.positive_pct || 0}% of total`;
            if (negPct) negPct.textContent = `${stats.negative_pct || 0}% of total`;
        }

        if (distributionChartInstance) {
            distributionChartInstance.data.datasets[0].data = [
                stats.positive || 0,
                stats.negative || 0,
                stats.neutral || 0
            ];
            if (stats.total > 0 && distributionChartInstance.options.plugins.title) {
                distributionChartInstance.options.plugins.title.text = 'Session Sentiment Breakdown';
            }
            distributionChartInstance.update();
        }

        if (stats && typeof stats.domain_distribution !== 'undefined') {
            renderDomainAnalytics(stats.domain_distribution);
        }
    } catch (e) {
        console.warn('Chart refresh warning:', e);
    }
}
