"""
benchmark_standard_length.py
Standard Length Benchmark (10 essays, 250-320 words) on Education Topic.
Includes standard band progressions (5.0 - 8.0) and adversarial edge-cases:
- S04: Overuse of mechanical discourse markers (Pseuso-cohesion)
- S06: High linguistic proficiency but Partially Off-Topic
- S10: Band 8.0 depth & vocabulary with isolated grammatical slips
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

BENCHMARK_ESSAYS = [
    {
        "id": "S01",
        "target": 5.0,
        "type": "Standard Band 5.0",
        "notes": "Limited elaboration, repetitive language, basic syntax, meets word count.",
        "essay": (
            "Nowadays, tertiary education is very significant for all students around the world. "
            "Some individuals believe that students should choose any subject they like at university, "
            "but other people argue that they must only learn practical courses like science and IT. "
            "In my essay, I will discuss both perspectives and give my point of view.\n\n"
            "On the one hand, learning favourite subjects gives students high motivation. When students "
            "like what they study, they will feel happy and enjoy attending classes every single day. "
            "For instance, if someone is keen on drawing or painting, they can become a creative artist "
            "or an art teacher. They can make beautiful paintings for the public. If they are forced "
            "to study boring courses, they might get low marks and feel stressed.\n\n"
            "On the other hand, learning technical subjects helps young people get employment easily. "
            "Today, the economy needs many engineers, doctors, and software developers. These professions "
            "usually provide stable work and high salaries for fresh graduates. Therefore, people can "
            "feed their families and pay for their living costs without worrying about debt. Science "
            "is also useful to make new machines and improve public health.\n\n"
            "In conclusion, although studying technical subjects brings high income, I think students "
            "should have the right to select their favorite degrees. Universities should give them advice "
            "so they can find good jobs after they finish their tertiary studies."
        )
    },
    {
        "id": "S02",
        "target": 5.5,
        "type": "Standard Band 5.5",
        "notes": "Adequate coverage, straightforward ideas, conventional connectors, mix of simple/complex sentences.",
        "essay": (
            "The question of whether university undergraduates ought to pursue their personal interests "
            "or focus strictly on vocationally useful courses is an important educational dilemma. "
            "While utilitarian courses offer clear economic benefits, I believe that personal inclination "
            "should still play a central role in academic choices.\n\n"
            "Proponents of practical subjects argue that higher education should directly prepare students "
            "for the competitive labor market. Modern industries increasingly depend on scientific breakthroughs "
            "and digital technologies, meaning that qualifications in computer science and engineering are in high "
            "demand. Graduates with these specific degrees are less likely to face unemployment after graduation "
            "and can contribute immediately to technological innovation in their country. From this viewpoint, "
            "public resources invested in university education should yield clear economic returns.\n\n"
            "However, compelling students to study fields solely because of commercial demand can create serious "
            "drawbacks. Academic success requires continuous effort and dedication, which are difficult to maintain "
            "if a student lacks genuine enthusiasm for the discipline. Furthermore, non-technical subjects like "
            "literature, philosophy, and history develop valuable abilities, including critical thinking, clear "
            "communication, and cultural awareness. These versatile skills are applicable across diverse industries "
            "and remain fundamental to a civilized society.\n\n"
            "In conclusion, although practical degrees provide reliable career pathways, I contend that students "
            "should be allowed to select disciplines aligned with their interests. Institutions should assist them "
            "by offering practical career counseling alongside their chosen academic programs."
        )
    },
    {
        "id": "S03",
        "target": 6.0,
        "type": "Standard Band 6.0",
        "notes": "Clear position throughout, relevant main ideas with supporting evidence, some complex structures.",
        "essay": (
            "In contemporary society, whether tertiary students should have complete autonomy over their major "
            "or be restricted to practical fields like science and technology remains a topic of intense discussion. "
            "Although focusing on practical disciplines provides clear career advantages, I would argue that "
            "individuals should retain the freedom to pursue subjects that match their passions.\n\n"
            "There are valid reasons why some people advocate for utilitarian education. The contemporary job "
            "market is rapidly evolving, with automation and digital transformation driving immense demand for "
            "skilled engineers, data analysts, and medical specialists. By directing undergraduates into these "
            "fields, nations can bridge critical labor shortages and accelerate economic development. Furthermore, "
            "students who graduate with technological expertise generally enjoy superior job security and higher "
            "starting salaries, which reduces the financial burden of student loans.\n\n"
            "Nonetheless, limiting university education strictly to commercial needs overlooks the intrinsic purpose "
            "of higher learning. Academic excellence is closely correlated with personal motivation; forcing students "
            "into technical areas for which they possess neither aptitude nor interest often results in academic "
            "disillusionment and high dropout rates. Moreover, disciplines in the arts and humanities foster essential "
            "qualities such as empathetic understanding, ethical analysis, and creative problem-solving, which are "
            "equally indispensable for societal well-being.\n\n"
            "To conclude, while prioritizing science and technology is beneficial for national productivity, "
            "I firmly believe that students should have the autonomy to choose their educational pathways. "
            "Higher education institutions should support this autonomy by integrating practical career modules "
            "into all degree programs."
        )
    },
    {
        "id": "S04",
        "target": 6.0,
        "type": "Adversarial: Overused Mechanical Linkers (Weak Coherence)",
        "notes": "Starts almost every sentence with formal transition words; mechanical cohesion; repetitive structure.",
        "essay": (
            "Firstly, university education is an essential factor for young citizens. However, people have disparate "
            "opinions regarding academic majors. Furthermore, some individuals argue for free choice. In addition, "
            "others believe that practical disciplines like technology are necessary. In this essay, I will analyze "
            "both viewpoints comprehensively.\n\n"
            "On the one hand, studying science is advantageous. Moreover, modern industries need technical specialists. "
            "For example, IT companies hire software engineers regularly. Consequently, graduates obtain substantial salaries. "
            "Therefore, unemployment rates can be mitigated effectively. In addition, scientific developments support national "
            "infrastructure. Hence, governments encourage technical degrees.\n\n"
            "On the other hand, personal interest is paramount. Furthermore, motivation is required for academic achievements. "
            "To illustrate, an art student works diligently because of genuine enthusiasm. Conversely, forcing students into "
            "science causes severe psychological distress. Additionally, humanities cultivate cultural comprehension. "
            "Thus, non-scientific disciplines are socially advantageous.\n\n"
            "In conclusion, practical disciplines yield tangible employment opportunities. Nevertheless, personal liberty "
            "is exceedingly crucial. Overall, universities should accommodate both dimensions systematically."
        )
    },
    {
        "id": "S05",
        "target": 6.5,
        "type": "Standard Band 6.5",
        "notes": "Logically organized, good progression, clear topic sentences, varied vocabulary with good collocations.",
        "essay": (
            "The question of whether university entrants should be encouraged to pursue their personal interests "
            "or steered towards disciplines that fulfill economic demands, such as science and technology, is subject "
            "to ongoing debate. While technical degrees indisputably offer robust employment prospects, I believe "
            "that academic freedom is vital for both individual fulfillment and well-rounded societal progress.\n\n"
            "Advocates of prioritizing utilitarian subjects emphasize the tangible requirements of modern economies. "
            "As industries undergo rapid digital modernization, specialized knowledge in computer programming, "
            "biotechnology, and engineering has become indispensable. Directing university curricula towards these "
            "critical sectors ensures that graduates transition smoothly into the labor force, thereby minimizing "
            "graduate unemployment and stimulating national productivity. For students seeking upward socioeconomic "
            "mobility, choosing an in-demand technical discipline represents a rational and secure investment.\n\n"
            "Nevertheless, a purely functional view of higher education fails to acknowledge the multifaceted nature "
            "of human talent. True innovation rarely stems from technical proficiency alone; it thrives at the "
            "intersection of technology, ethics, and human experience. Disciplines such as philosophy, history, "
            "and literature nurture critical thinking, qualitative reasoning, and effective communication—skills that "
            "enable leaders to address complex global dilemmas responsibly. Furthermore, coercing students into "
            "unwanted technical courses frequently engenders disengagement, whereas intrinsic passion consistently "
            "drives profound academic and professional breakthroughs.\n\n"
            "In conclusion, while the economic imperative of science and technology cannot be overlooked, university "
            "education should not be reduced to vocational training. Providing students with the liberty to choose "
            "their subjects, while equipping them with transferable career competencies, is the most balanced approach."
        )
    },
    {
        "id": "S06",
        "target": 5.5,
        "type": "Adversarial: Off-Topic / Tangential Response",
        "notes": "Flawless grammar and advanced vocabulary, but strays heavily into tuition fees and university funding.",
        "essay": (
            "Higher education has undergone monumental transformations over recent decades, raising pressing dilemmas "
            "regarding pedagogical priorities and financial sustainability. While technological education is considered "
            "advantageous, the debate surrounding academic selection is inextricably linked to university tuition fees "
            "and government fiscal allocations.\n\n"
            "Primarily, university education has become exorbitantly expensive in contemporary economies. Many tertiary "
            "institutions have drastically inflated their semester fees, forcing young scholars to incur colossal student "
            "debts before entering the workforce. Consequently, government authorities ought to subsidize tertiary tuition "
            "by channeling fiscal reserves directly into public universities. When campus infrastructure and dormitory "
            "facilities are thoroughly upgraded, undergraduates can immerse themselves in rigorous scholarly inquiry "
            "without constant pecuniary anxiety. Scholarships should also be distributed equitably to marginalized demographics.\n\n"
            "Furthermore, international student admissions generate substantial revenue streams for Western universities. "
            "Institutions in Australia and the United Kingdom rely extensively on exorbitant overseas tuition rates to "
            "finance their operational expenditures and professorial salaries. Therefore, academic administrators must "
            "establish transparent budgetary auditing mechanisms to prevent institutional fiscal mismanagement. Without "
            "comprehensive financial oversight, university standards will inevitably deteriorate.\n\n"
            "In conclusion, resolving fiscal disparities and alleviating tuition inflation are of paramount importance "
            "for higher educational institutions. Only when tertiary education is adequately funded can universities "
            "deliver high-caliber academic experiences to future generations."
        )
    },
    {
        "id": "S07",
        "target": 7.0,
        "type": "Standard Band 7.0",
        "notes": "Well-developed, nuanced position, precise academic vocabulary, varied complex sentences with high accuracy.",
        "essay": (
            "The overarching purpose of higher education has long been a subject of contention, specifically whether "
            "undergraduates should enjoy unfettered freedom in choosing their disciplines or be channeled into "
            "economically lucrative areas such as STEM (Science, Technology, Engineering, and Mathematics). Although "
            "focusing on market-driven subjects brings undeniable economic advantages, I contend that preserving "
            "individual academic autonomy is essential for sustainable societal development.\n\n"
            "On the one hand, the argument for prioritizing utilitarian subjects is anchored in pragmatic economic "
            "realities. In an era marked by artificial intelligence and automated systems, specialized proficiency "
            "in technical fields is paramount for national competitiveness. Nations with a robust surplus of software "
            "engineers, biomedical researchers, and data analysts can spearhead innovation, attract foreign direct "
            "investment, and maintain industrial resilience. For individuals, acquiring marketable technical skills "
            "significantly mitigates the risk of career obsolescence, offering predictable employment trajectories and "
            "financial security in an uncertain economic landscape.\n\n"
            "On the other hand, restricting academic pursuits solely to commercially viable fields represents a "
            "myopic understanding of human capability and societal enrichment. Sustainable progress relies not merely "
            "on technological execution, but equally on ethical introspection, creative imagination, and cultural "
            "preservation—facets nurtured primarily through the humanities and social sciences. Furthermore, students "
            "who are intrinsically motivated to study disciplines like history, philosophy, or literature often develop "
            "exceptional capacities for analytical synthesis and nuanced communication, rendering them highly versatile "
            "leaders. Forcing individuals into technical trajectories contrary to their natural aptitudes inevitably "
            "breeds professional dissatisfaction and undermines long-term productivity.\n\n"
            "In conclusion, while the strategic promotion of science and technology is vital for economic progress, "
            "it should not eclipse academic freedom. Universities should honor personal choice while offering robust "
            "interdisciplinary curricula that empower graduates to thrive across diverse professional domains."
        )
    },
    {
        "id": "S08",
        "target": 7.5,
        "type": "Standard Band 7.5",
        "notes": "Nuanced perspective, sophisticated collocations, smooth transitions, high grammatical flexibility.",
        "essay": (
            "The perpetual tension between educational idealism and labor-market pragmatism is vividly manifested in "
            "the debate over university curricula. While proponents of utilitarianism advocate directing undergraduates "
            "towards vocationally viable fields such as science and technology, others champion the principle of "
            "unrestricted academic discretion. Although technical expertise is indispensable to contemporary economic "
            "vitality, I firmly maintain that preserving student autonomy remains paramount for comprehensive societal "
            "and intellectual progress.\n\n"
            "The rationale for prioritizing scientific and technological disciplines is undeniably formidable. In an "
            "increasingly digitized global economy, technological literacy forms the bedrock of national productivity. "
            "Sectors such as machine learning, renewable energy engineering, and biotechnology dictate geopolitical "
            "influence, requiring a continuous influx of specialized talent. Directing university cohorts towards these "
            "strategic domains effectively remedies structural employment deficits and safeguards young graduates against "
            "economic vulnerability. From a purely fiscal perspective, aligning academic output with industrial demand "
            "maximizes the return on public and private educational investments.\n\n"
            "Nevertheless, treating higher education merely as a conveyor belt for corporate recruitment is profoundly "
            "reductive. Human flourishing and democratic governance rely equally on disciplines that interrogate ethical "
            "dilemmas, critique societal paradigms, and preserve historical memory. Subjects such as philosophy, sociology, "
            "and comparative literature cultivate intellectual agility, hermeneutic depth, and persuasive rhetoric—attributes "
            "that cannot be mechanized. Crucially, empirical evidence suggests that intrinsic motivation is the primary "
            "catalyst for extraordinary achievement; compelling scholars into fields uncongenial to their temperaments "
            "fosters mediocrity rather than excellence.\n\n"
            "Ultimately, a forward-looking society must reject this false dichotomy between utility and passion. "
            "Higher education institutions should uphold the sovereignty of academic choice while embedding transferable "
            "competencies into non-technical degree pathways, thereby cultivating both specialized innovators and "
            "enlightened citizens."
        )
    },
    {
        "id": "S09",
        "target": 8.0,
        "type": "Standard Band 8.0",
        "notes": "Sophisticated argumentation, seamless coherence, precise academic register, full grammatical control.",
        "essay": (
            "The philosophical debate regarding the fundamental mandate of tertiary education frequently oscillates "
            "between two contrasting doctrines: humanistic self-actualization and macroeconomic utilitarianism. While "
            "technocratic arguments advocate channeling students exclusively into practical disciplines such as STEM, "
            "I contend that safeguarding academic autonomy is vital not only for individual fulfillment but also for "
            "maintaining an intellectually dynamic and resilient civilization.\n\n"
            "The impetus behind prioritizing science and technology is undeniably substantiated by contemporary economic "
            "imperatives. In an era increasingly defined by computational paradigms, artificial intelligence, and ecological "
            "transitions, industrial competitiveness hinges on technical sophistication. A concentrated output of STEM "
            "graduates addresses pressing societal bottlenecks—from decarbonizing energy grids to engineering epidemiological "
            "interventions—while insulating individuals from structural underemployment. Given the substantial capital "
            "expenditure involved in higher education, optimizing degree pathways according to empirical labor forecasts "
            "appears pragmatically sound.\n\n"
            "However, subordinating pedagogical freedom entirely to transient market forces embodies an insidious form "
            "of instrumentalism. True scientific breakthroughs are rarely produced by coerced practitioners; rather, "
            "epochal discoveries emerge from uninhibited intellectual curiosity and intrinsic devotion. Furthermore, a "
            "civilization that disparages the humanities in favor of vocational expedience invites profound cultural and "
            "philosophical impoverishment. Disciplines such as moral philosophy, jurisprudence, and literary criticism "
            "provide the indispensable ethical scaffolds and interpretive nuance without which technological power "
            "becomes erratic and self-destructive. Transferable intellectual competencies—most notably dialectical reasoning "
            "and epistemological skepticism—are predominantly forged within these liberal arts traditions.\n\n"
            "In conclusion, while the strategic promotion of STEM disciplines is economically indispensable, it must not "
            "degenerate into bureaucratic prescription. Universities fulfill their noblest societal function when they "
            "preserve the sanctity of individual intellectual pursuit, simultaneously fostering interdisciplinary synthesis "
            "between technological capability and humanistic wisdom."
        )
    },
    {
        "id": "S10",
        "target": 8.0,
        "type": "Adversarial: Band 8.0 Depth with 2 Isolated Typo/Grammar Slips",
        "notes": "Band 8.0 lexical and conceptual sophistication, but contains 2 minor slips ('a adaptable', 'criteria was').",
        "essay": (
            "The philosophical discourse concerning the primary remit of higher education routinely bifurcates between "
            "individual intellectual sovereignty and systemic economic utility. While certain policy analysts assert that "
            "undergraduates should strictly pursue high-utility vocations such as engineering and computer science, I "
            "maintain that preserving unrestricted discipline choice is imperative for sustained sociocultural and "
            "scientific vitality.\n\n"
            "Advocates of vocationally tethered education point persuasively to structural economic exigencies. As digital "
            "technologies continuously disrupt conventional employment paradigms, the imperative for advanced technical "
            "literacy escalates exponentially. Directing students toward high-demand sectors mitigates graduate underemployment "
            "and furnishes sovereign states with the specialized human capital essential for national competitiveness. From a "
            "utilitarian calculus, aligning academic admissions with verified industrial deficits maximizes public investment "
            "and cushions graduates against macroeconomic volatility.\n\n"
            "Yet, this narrowly technocratic vision neglects the fundamental mechanisms through which excellence is fostered. "
            "Compelling undergraduates into predetermined fields irrespective of genuine affinity predictably generates "
            "pedagogical disengagement and sub-optimal mastery. In contrast, intrinsic motivation constitutes the decisive "
            "catalyst for pioneering research. Moreover, an exclusively utilitarian curriculum risks eviscerating the humanities, "
            "which nurture ethical acuity, dialectical reasoning, and civic literacy. These non-technical competencies remain "
            "indispensable when formulating public policy or adjudicating the moral ramifications of emerging technologies.\n\n"
            "In conclusion, while practical fields warrant institutional support, they should not be imposed as the sole "
            "legitimate avenue of academic inquiry. A higher education system that honors student autonomy while providing "
            "rigorous interdisciplinary exposure produces both a adaptable workforce and an enlightened citizenry."
        )
    }
]

def main():
    print("Loading TriSpecialistModel...")
    model = TriSpecialistModel.load("data/model_tri_specialist.pkl")

    results = []
    print("\nRunning Standard Length Benchmark (10 essays, 250-320 words)...\n")

    for item in BENCHMARK_ESSAYS:
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
            "type": item["type"],
            "rules": rules
        })

    # Summary table
    print("=" * 105)
    print(f"{'ID':<5} | {'Target':<6} | {'TR':<5} | {'CC':<5} | {'LR':<5} | {'GRA':<5} | {'Overall':<7} | {'Abs Err':<7} | {'Words':<5} | {'Category'}")
    print("-" * 105)
    for r in results:
        print(f"{r['id']:<5} | {r['target']:<6.1f} | {r['tr']:<5.1f} | {r['cc']:<5.1f} | {r['lr']:<5.1f} | {r['gra']:<5.1f} | {r['overall']:<7.1f} | {r['abs_err']:<7.1f} | {r['wc']:<5} | {r['type']}")
    print("=" * 105)

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

    print("\nSTANDARD LENGTH BENCHMARK METRICS (N = 10, Words = 250-320):")
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
