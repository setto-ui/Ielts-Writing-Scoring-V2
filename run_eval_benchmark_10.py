"""
run_eval_benchmark_10.py
Runs the 10 benchmark essays through IELTS Writing Scorer v2.2 (TriSpecialistModel).
Outputs full comparison table with reference vs predicted scores, error breakdown, and metrics.
"""

import sys
import numpy as np

# Encoding fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluate import FEATURE_KEYS, extract_all_features, features_to_array
from scorer.normalizer import normalize_features
from scorer.two_stage_model import TriSpecialistModel
from scorer.scorer import score_essay, round_to_ielts_band
from scorer.constraint import apply_constraints

PROMPT = (
    "Some people believe that university students should study whatever they like, "
    "while others think they should only study subjects that will be useful in the future, "
    "such as science and technology. Discuss both views and give your own opinion."
)

BENCHMARK_DATA = [
    {
        "id": "W01",
        "target": 4.0,
        "target_sub": {"TR": 4.0, "CC": 4.0, "LR": 4.0, "GRA": 4.0},
        "essay": (
            "I think students should study what they like because they can be happy. "
            "Science is good for future and people need jobs. But some students like history and art. "
            "They can study it too. If they study something they like, they work hard. "
            "If they study science, they get money. I think students choose what they want. "
            "University is important and students should study."
        )
    },
    {
        "id": "W02",
        "target": 4.5,
        "target_sub": {"TR": 4.5, "CC": 4.5, "LR": 4.5, "GRA": 4.5},
        "essay": (
            "Some people think students should choose their subjects, but other people think science is better. "
            "I agree students choose what they like.Science and technology are useful because people can get a job. "
            "For example, computer jobs are popular now. Many companies need workers. "
            "This is good for students and the country.However, students like other subjects. "
            "Art and history are interesting. Students can learn about the past and make pictures. "
            "They can be happy when studying. They should not only study science.In conclusion, "
            "I think students should choose their subjects because they know what they like."
        )
    },
    {
        "id": "W03",
        "target": 5.0,
        "target_sub": {"TR": 5.0, "CC": 5.0, "LR": 5.0, "GRA": 5.0},
        "essay": (
            "There are different opinions about what university students should study. "
            "Some people believe that students should choose their favourite subjects, while others think "
            "that useful subjects are more important. I think both ideas are reasonable.On the one hand, "
            "science and technology can help students find jobs after graduation. Nowadays, many companies "
            "need engineers and IT workers. If students study these areas, they may have more chances to work "
            "and earn money. This is important because life is expensive.On the other hand, students should "
            "study subjects that they enjoy. When they like a subject, they can study harder and understand "
            "it better. For example, a student who likes history may become a good teacher. Art is also "
            "useful for society.In my opinion, students should choose their subjects, but they should "
            "also think about jobs. Universities can give advice to help them make a good decision."
        )
    },
    {
        "id": "W04",
        "target": 5.5,
        "target_sub": {"TR": 5.5, "CC": 5.5, "LR": 5.5, "GRA": 5.5},
        "essay": (
            "People have different views about whether university students should follow their interests "
            "or select subjects with better future value. Although both choices have benefits, I believe "
            "personal preference should be considered alongside employment prospects.Studying science "
            "and technology may help graduates find work more easily. Many businesses require workers with "
            "technical knowledge, especially in areas such as computing and engineering. These jobs can offer "
            "stable income, so students who choose practical courses may be less worried about unemployment. "
            "However, not everyone is suitable for these fields, and forcing students to study them could "
            "lead to poor performance.Choosing a subject based on interest can make education more enjoyable. "
            "Students who are motivated may spend more time learning and develop stronger skills. "
            "For example, someone interested in languages could become a translator or teacher. "
            "Nevertheless, interest alone may not guarantee a job, so students should research the career "
            "options before making a decision.Overall, universities should allow students to choose freely "
            "while providing information about employment and career development."
        )
    },
    {
        "id": "W05",
        "target": 6.0,
        "target_sub": {"TR": 6.0, "CC": 6.0, "LR": 6.0, "GRA": 6.0},
        "essay": (
            "The choice of university subjects has become an important issue, as some people favour "
            "personal interests while others prioritize future employment. Both views are understandable, "
            "and I believe students should make this decision by considering their abilities as well as "
            "the job market.Supporters of science and technology argue that these disciplines provide "
            "practical skills. As many industries are becoming digital, graduates with technical knowledge "
            "may find a wider range of opportunities. In addition, these professions can contribute to "
            "economic growth by developing new products and improving services. This is particularly "
            "important in countries where skilled workers are in short supply.However, choosing a subject "
            "simply because it is in demand can be a mistake. Students who have little interest in their "
            "courses may struggle to stay motivated, which can affect their performance. By contrast, "
            "those who study a field they enjoy are more likely to invest time in developing expertise. "
            "For instance, a student with a strong interest in psychology may become a capable counsellor "
            "or researcher.In my view, universities should not force students into particular disciplines. "
            "Instead, they should offer career advice and opportunities to gain practical experience, "
            "helping students connect their interests with realistic employment options."
        )
    },
    {
        "id": "W06",
        "target": 6.5,
        "target_sub": {"TR": 6.5, "CC": 6.5, "LR": 6.5, "GRA": 6.5},
        "essay": (
            "Whether undergraduates should pursue their preferred disciplines or concentrate on subjects "
            "with strong labour-market demand is widely debated. While technical fields can improve "
            "employability, I believe academic choice should reflect both personal aptitude and practical "
            "considerations.Science and technology degrees are often associated with promising career "
            "prospects. As businesses increasingly rely on digital systems, graduates in engineering, "
            "computing and related areas may be able to access a wider range of vacancies. Their expertise "
            "can also support innovation, productivity and economic growth. For students who prioritize "
            "financial independence, these advantages are difficult to ignore.Nevertheless, employment "
            "demand should not be the sole factor guiding a student's decision. A person who lacks interest "
            "or ability in a technical subject may find it difficult to perform well, even if the qualification "
            "is marketable. Conversely, students who are deeply engaged in humanities or creative disciplines "
            "can develop transferable skills, including communication, research and analytical thinking. "
            "These competencies are valuable across many professions.I would therefore favour a system in "
            "which students retain the freedom to choose their majors, while universities provide reliable "
            "labour-market information, internships and opportunities to acquire complementary skills. "
            "This would allow graduates to pursue meaningful work without overlooking economic realities."
        )
    },
    {
        "id": "W07",
        "target": 7.0,
        "target_sub": {"TR": 7.0, "CC": 7.0, "LR": 7.0, "GRA": 7.0},
        "essay": (
            "The question of whether university students should pursue their interests or select disciplines "
            "with clear economic value has become increasingly relevant. Although science-related degrees "
            "can lead to attractive employment prospects, I would argue that students should retain "
            "considerable freedom over their academic choices.Those who advocate prioritising practical "
            "subjects point to the changing demands of the labour market. Industries such as software, "
            "healthcare and engineering require specialised knowledge, and graduates in these areas may "
            "benefit from relatively strong demand. This can reduce the risk of unemployment and enable "
            "individuals to become financially independent sooner. At a broader level, a workforce with "
            "technical expertise can help a country remain competitive.However, employment prospects alone "
            "are an insufficient basis for choosing a degree. Academic success depends partly on sustained "
            "motivation, which is more likely when students are genuinely interested in what they study. "
            "Moreover, disciplines such as history, languages and the arts cultivate skills that are not "
            "confined to a single occupation, including interpretation, communication and creative "
            "problem-solving. These contributions may be less immediately measurable than technological "
            "advances, but they remain socially valuable.In my view, universities should not dictate "
            "students' choices according to short-term labour-market trends. Instead, they should help "
            "applicants make informed decisions through career counselling and work-placement opportunities. "
            "This approach respects individual preferences while acknowledging the practical need to prepare "
            "for employment."
        )
    },
    {
        "id": "W08",
        "target": 7.5,
        "target_sub": {"TR": 7.5, "CC": 7.5, "LR": 7.5, "GRA": 7.5},
        "essay": (
            "The debate over whether undergraduates should follow their intellectual interests or prioritise "
            "disciplines with greater economic utility reflects a broader tension between personal fulfilment "
            "and workforce needs. While degrees in science and technology can offer tangible career advantages, "
            "I believe students should be free to choose their fields, provided they make informed decisions."
            "The case for prioritising technical disciplines is compelling. As automation and digitalisation "
            "reshape industries, employers increasingly seek graduates with expertise in areas such as data "
            "science, engineering and cybersecurity. Such qualifications may improve employability and, in "
            "some cases, provide access to well-paid positions. A steady supply of technically skilled workers "
            "can also strengthen a country's capacity for innovation and productivity.Nevertheless, the value "
            "of higher education cannot be measured solely by graduates' salaries or immediate employment outcomes. "
            "Students who pursue subjects aligned with their interests are often more engaged, which can lead to "
            "deeper learning and stronger performance. Humanities and arts programmes, for example, foster "
            "cultural understanding, critical analysis and communication—abilities that are transferable to a "
            "wide range of careers. Restricting academic choices to supposedly 'useful' disciplines could also "
            "leave society with a narrow conception of knowledge.Universities should therefore preserve academic "
            "choice while ensuring that students understand the likely costs, opportunities and career pathways "
            "associated with different degrees. Through internships, career guidance and transferable-skills "
            "training, institutions can help students bridge the gap between personal ambition and professional realities."
        )
    },
    {
        "id": "W09",
        "target": 8.0,
        "target_sub": {"TR": 8.0, "CC": 8.0, "LR": 8.0, "GRA": 8.0},
        "essay": (
            "Whether students should be free to pursue any academic discipline or be steered towards fields with "
            "demonstrable economic value is a question of both individual autonomy and public policy. Although "
            "science and technology are indispensable to economic progress, I contend that university choices "
            "should primarily reflect students' aptitudes and interests, informed by realistic career expectations."
            "The argument for prioritising technical education is understandable. Digital transformation has "
            "created demand for expertise in fields such as computing, engineering and biotechnology, and "
            "graduates with these skills may enjoy stronger employment prospects. Public investment in such "
            "programmes can also yield substantial returns through innovation, improved infrastructure and more "
            "efficient services. From this perspective, directing students towards high-demand disciplines may "
            "benefit both individuals and society.Yet a narrowly utilitarian approach to higher education risks "
            "overlooking the conditions under which people excel. Students are more likely to develop expertise "
            "when their studies align with their abilities and intrinsic motivation; forcing them into a fashionable "
            "field may produce disengagement rather than a competent workforce. Furthermore, the humanities and arts "
            "contribute in ways that are not easily captured by salary statistics, from preserving cultural heritage "
            "to strengthening ethical reasoning and communication. These capacities are also relevant in technical "
            "workplaces, where complex problems rarely have purely technical solutions.A more defensible policy "
            "would combine freedom of choice with informed guidance. Universities could publish transparent "
            "employment data, offer placements and encourage interdisciplinary study, allowing students to pursue "
            "their interests while acquiring skills that make them employable. This would avoid treating current "
            "labour-market demand as a permanent measure of a subject's worth.In conclusion, practical disciplines "
            "deserve recognition for their economic contribution, but they should not be imposed as the only worthwhile "
            "options. Academic choice, supported by reliable career information, is more likely to produce both "
            "fulfilled graduates and a adaptable workforce."
        )
    },
    {
        "id": "W10",
        "target": 8.5,
        "target_sub": {"TR": 8.5, "CC": 8.5, "LR": 8.5, "GRA": 8.5},
        "essay": (
            "The tension between academic freedom and the economic priorities of higher education is often "
            "framed as a choice between personal fulfilment and practical utility. While degrees in science and "
            "technology can equip graduates for sectors experiencing rapid growth, I believe students should be "
            "free to pursue disciplines suited to their aptitudes, with career guidance helping them weigh the consequences."
            "The rationale for favouring technical subjects is not merely financial. Scientific and technological "
            "expertise underpins advances in medicine, infrastructure and digital services, and shortages in these "
            "areas can constrain national development. Graduates with specialised skills may also transition more "
            "readily into occupations where demand is robust. For students who value financial security, these "
            "considerations understandably carry considerable weight.However, employment forecasts are imperfect "
            "predictors of the opportunities graduates will encounter several years later. Steering large numbers "
            "of students towards currently fashionable disciplines could eventually produce oversupply, while "
            "neglecting fields whose social value is less readily quantified. More fundamentally, sustained achievement "
            "depends on more than market demand: intellectual curiosity, aptitude and commitment shape whether "
            "individuals become capable professionals. The humanities and creative arts, for instance, nurture "
            "interpretation, cultural literacy and communication, qualities that complement rather than compete with "
            "technical expertise.Universities should therefore neither disregard economic realities nor reduce "
            "education to workforce preparation. They can preserve students' freedom while making choices more "
            "informed through transparent outcome data, work placements and interdisciplinary programmes. "
            "Such measures would allow individuals to pursue meaningful study and develop adaptable skills without "
            "assuming that today's labour-market priorities will remain unchanged.Ultimately, the most effective "
            "higher-education system is not one that dictates a narrow set of 'useful' subjects, but one that equips "
            "students to make thoughtful choices and contribute in diverse ways. This balance recognises both the "
            "economic role of universities and their broader intellectual and social purpose."
        )
    }
]

def main():
    print("Loading TriSpecialistModel...")
    model = TriSpecialistModel.load("data/model_tri_specialist.pkl")

    results = []
    print("\nRunning evaluation on 10 benchmark essays...\n")

    for item in BENCHMARK_DATA:
        essay = item["essay"]
        raw = extract_all_features(essay, PROMPT)
        norm = normalize_features(raw)
        X = features_to_array(norm, FEATURE_KEYS).reshape(1, -1)

        ml_raw = model.predict(X)
        ml_preds = {
            k: float(v[0]) if hasattr(v, "__len__") else float(v)
            for k, v in ml_raw.items()
        }

        score_result = score_essay(
            normalized_features=norm,
            raw_features=raw,
            ml_predictions=ml_preds,
        )
        constrained, rules = apply_constraints(score_result["scores"], raw)
        overall_pred = round_to_ielts_band(float(constrained.get("overall", ml_preds["overall"])))
        sub_preds = {k: round_to_ielts_band(v) for k, v in constrained.items()}

        target = item["target"]
        abs_err = abs(overall_pred - target)
        err = overall_pred - target
        wc = int(raw.get("word_count", 0))

        results.append({
            "id": item["id"],
            "target": target,
            "overall": overall_pred,
            "tr": sub_preds.get("task_response", overall_pred),
            "cc": sub_preds.get("coherence", overall_pred),
            "lr": sub_preds.get("lexical", overall_pred),
            "gra": sub_preds.get("grammar", overall_pred),
            "abs_err": abs_err,
            "err": err,
            "wc": wc,
            "rules": rules
        })

    # Summary table
    print("=" * 95)
    print(f"{'ID':<5} | {'Target':<6} | {'TR':<5} | {'CC':<5} | {'LR':<5} | {'GRA':<5} | {'Overall':<7} | {'Abs Err':<7} | {'Words':<5} | {'Rules'}")
    print("-" * 95)
    for r in results:
        rules_str = ", ".join(r["rules"]) if r["rules"] else "none"
        print(f"{r['id']:<5} | {r['target']:<6.1f} | {r['tr']:<5.1f} | {r['cc']:<5.1f} | {r['lr']:<5.1f} | {r['gra']:<5.1f} | {r['overall']:<7.1f} | {r['abs_err']:<7.1f} | {r['wc']:<5} | {rules_str}")
    print("=" * 95)

    # Metrics
    targets = np.array([r["target"] for r in results])
    preds = np.array([r["overall"] for r in results])
    abs_errors = np.abs(preds - targets)
    errors = preds - targets

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    exact_acc = float(np.mean(abs_errors == 0.0) * 100)
    acc_05 = float(np.mean(abs_errors <= 0.5) * 100)
    acc_10 = float(np.mean(abs_errors <= 1.0) * 100)
    over_score_rate = float(np.mean(errors > 0.0) * 100)
    under_score_rate = float(np.mean(errors < 0.0) * 100)
    bias = float(np.mean(errors))

    print("\nBENCHMARK SUMMARY METRICS (N = 10):")
    print(f"  • MAE (Mean Absolute Error)     : {mae:.3f}")
    print(f"  • RMSE                          : {rmse:.3f}")
    print(f"  • Exact Match Accuracy          : {exact_acc:.1f}%")
    print(f"  • Within +-0.5 Band Accuracy    : {acc_05:.1f}%")
    print(f"  • Within +-1.0 Band Accuracy    : {acc_10:.1f}%")
    print(f"  • Over-scoring Rate             : {over_score_rate:.1f}%")
    print(f"  • Under-scoring Rate            : {under_score_rate:.1f}%")
    print(f"  • Directional Bias (Pred - Ref) : {bias:+.3f}")

if __name__ == "__main__":
    main()
