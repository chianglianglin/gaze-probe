/* speech_core.js — 發音比對的核心邏輯（不碰畫面、不碰模型載入）
   做法：模型（wav2vec2 音素辨識，CTC）自己輸出的音素標籤不準，所以不直接用。
   改成把「目標句子的音素」強制對齊到聲音上（forced alignment），再做兩種檢查：
   1. 長音／促音：量每一拍的長度，和示範音比（先用整句語速校正）。
   2. 清濁音：把目標裡的清音換成對應的濁音（或反過來），比較哪一個版本的機率高。 */
(function(){
"use strict";
const R1={あ:"a",い:"i",う:"u",え:"e",お:"o",か:"ka",き:"ki",く:"ku",け:"ke",こ:"ko",さ:"sa",し:"shi",す:"su",せ:"se",そ:"so",た:"ta",ち:"chi",つ:"tsu",て:"te",と:"to",な:"na",に:"ni",ぬ:"nu",ね:"ne",の:"no",は:"ha",ひ:"hi",ふ:"fu",へ:"he",ほ:"ho",ま:"ma",み:"mi",む:"mu",め:"me",も:"mo",や:"ya",ゆ:"yu",よ:"yo",ら:"ra",り:"ri",る:"ru",れ:"re",ろ:"ro",わ:"wa",を:"o",が:"ga",ぎ:"gi",ぐ:"gu",げ:"ge",ご:"go",ざ:"za",じ:"ji",ず:"zu",ぜ:"ze",ぞ:"zo",だ:"da",ぢ:"ji",づ:"zu",で:"de",ど:"do",ば:"ba",び:"bi",ぶ:"bu",べ:"be",ぼ:"bo",ぱ:"pa",ぴ:"pi",ぷ:"pu",ぺ:"pe",ぽ:"po"};
const R2={きゃ:"kya",きゅ:"kyu",きょ:"kyo",しゃ:"sha",しゅ:"shu",しょ:"sho",ちゃ:"cha",ちゅ:"chu",ちょ:"cho",ちぇ:"che",にゃ:"nya",にゅ:"nyu",にょ:"nyo",ひゃ:"hya",ひゅ:"hyu",ひょ:"hyo",みゃ:"mya",みゅ:"myu",みょ:"myo",りゃ:"rya",りゅ:"ryu",りょ:"ryo",ぎゃ:"gya",ぎゅ:"gyu",ぎょ:"gyo",じゃ:"ja",じゅ:"ju",じょ:"jo",じぇ:"je",びゃ:"bya",びゅ:"byu",びょ:"byo",ぴゃ:"pya",ぴゅ:"pyu",ぴょ:"pyo",ふぁ:"fa",ふぃ:"fi",ふぇ:"fe",ふぉ:"fo",てぃ:"ti",でぃ:"di",うぃ:"wi",うぇ:"we"};
const k2h=s=>s.replace(/[ァ-ヶ]/g,c=>String.fromCharCode(c.charCodeAt(0)-0x60));
const VOW="aiueo",isV=p=>VOW.includes(p);
const VOICELESS=new Set(["k","s","sh","t","ch","ts","h","f","p","ky","hy","py"]);

/* 假名 → 目標音素 [{p, opt}]。長音寫成兩個相同母音，促音是 Q，撥音是 N；opt 表示常被無聲化、可以不出現。 */
function kanaToPhones(kana){
  const out=[];
  for(const w of k2h(kana).split(/[\s、。？?！!]+/)){
    if(!w)continue;
    if(w==="は"){out.push({p:"w"},{p:"a"});continue;}
    if(w==="へ"){out.push({p:"e"});continue;}
    for(let i=0;i<w.length;i++){
      const two=w.substr(i,2);let r;
      if(R2[two]){r=R2[two];i++;}
      else if(w[i]==="っ"){out.push({p:"Q"});continue;}
      else if(w[i]==="ん"){out.push({p:"N"});continue;}
      else if(w[i]==="ー"){const last=out[out.length-1];if(last&&isV(last.p)) out.push({p:last.p});continue;}
      else r=R1[w[i]];
      if(!r)continue;
      const v=r.slice(-1),c=r.slice(0,-1);
      if(c) out.push({p:c});
      const prev=out[out.length-1];
      if(!c&&v==="u"&&prev&&(prev.p==="o"||prev.p==="u")&&i>0) out.push({p:prev.p});   // おう→oo、うう→uu
      else out.push({p:v});
    }
  }
  for(let i=0;i<out.length;i++){
    const t=out[i];if(t.p!=="i"&&t.p!=="u")continue;
    const a=out[i-1],b=out[i+1];
    if(a&&VOICELESS.has(a.p)&&(!b||VOICELESS.has(b.p)||b.p==="Q")) t.opt=true;
  }
  return out;
}

/* 模型詞表裡的國際音標 → 簡化音素（只用來把詞表分組，以及顯示模型自己輸出了什麼） */
const MULTI=[["tɕ","ch"],["dʑ","j"],["tʃ","ch"],["dʒ","j"],["ts","ts"],["dz","z"]];
const SINGLE={a:"a","ɑ":"a","ɐ":"a","æ":"a",i:"i","ɪ":"i","ɨ":"i",u:"u","ɯ":"u","ʊ":"u","ʉ":"u",e:"e","ɛ":"e",o:"o","ɔ":"o","ə":"ə",
 k:"k","ɡ":"g",g:"g",s:"s","ɕ":"sh","ʃ":"sh",z:"z","ʑ":"j","ʒ":"j",t:"t",d:"d",n:"n","ɲ":"n","ɴ":"N","ŋ":"N",N:"N",m:"m",h:"h","ç":"h",x:"h","ɸ":"f",f:"f",b:"b","β":"b",v:"b",p:"p",j:"y","ɾ":"r",r:"r",l:"r","ɹ":"r","ɽ":"r","ɭ":"r",w:"w","ɰ":"w",q:"k",c:"k"};
function normToken(tok){
  let s=tok.normalize("NFD").replace(/[̀-ͯ]/g,"").replace(/[ᵝʰ\d.^"\[\]]/g,"");
  if(s.length>1&&/[ptkcqsɕ]h$/.test(s)) s=s.slice(0,-1);
  const out=[];
  while(s.length){
    let hit=false;
    for(const [a,b] of MULTI) if(s.startsWith(a)){out.push(b);s=s.slice(a.length);hit=true;break;}
    if(hit)continue;
    const ch=s[0];s=s.slice(1);
    if(ch==="ː"||ch===":"){const last=out[out.length-1];if(last==null)continue;if(isV(last)) out.push(last);else out.splice(out.length-1,0,"Q");continue;}
    if(ch==="ʲ"){out.push("y");continue;}
    if(ch==="ʔ")continue;
    out.push(SINGLE[ch]||ch);
  }
  return out;
}

/* ---------- 詞表分組：每個簡化音素對應模型詞表裡哪些符號 ---------- */
let vocab=null,sets=null;
function init(v){
  vocab=v;sets={};
  const add=(p,i)=>{(sets[p]=sets[p]||new Set()).add(i);};
  vocab.forEach((tok,i)=>{
    if(!tok||tok[0]==="<")return;
    const seq=normToken(tok).filter(x=>x!=="Q");
    if(seq.length&&seq.every(x=>x===seq[0])) add(seq[0],i);
  });
  const extra={a:["ʌ"],u:["ə"],r:["d"],N:["n","m"],f:["h"],g:["ɣ"],w:["ʋ"],e:["eɪ"],o:["oʊ"]};
  for(const [p,toks] of Object.entries(extra)) for(const t of toks){const i=vocab.indexOf(t);if(i>=0) add(p,i);}
}
const setOf=p=>sets[p]||(p.length===2&&p[1]==="y"?sets[p[0]]:null);

/* logits（frames × vocab）→ 每格的 log 機率。一格 = 20 毫秒。 */
function logSoftmax(dims,data){
  const F=dims[dims.length-2],S=dims[dims.length-1],ls=new Float32Array(F*S);
  for(let t=0;t<F;t++){
    let mx=-1e30;for(let v=0;v<S;v++) if(data[t*S+v]>mx) mx=data[t*S+v];
    let sum=0;for(let v=0;v<S;v++) sum+=Math.exp(data[t*S+v]-mx);
    const lz=mx+Math.log(sum);for(let v=0;v<S;v++) ls[t*S+v]=data[t*S+v]-lz;
  }
  return {F,S,ls};
}
/* 模型自己會輸出什麼（只供顯示） */
function greedy(L){
  const {F,S,ls}=L,toks=[];let prev=-1;
  for(let t=0;t<F;t++){
    let best=0;for(let v=1;v<S;v++) if(ls[t*S+v]>ls[t*S+best]) best=v;
    if(best===prev)continue;prev=best;
    const tok=vocab[best];if(tok&&tok[0]!=="<") toks.push(tok);
  }
  return toks;
}

/* ---------- 強制對齊 ---------- */
/* 促音本身沒有聲音，長音和短音是同一個音，所以對齊時把 Q 拿掉、連續相同母音合成一個，
   只在音素上記 q（前面有促音）、long（是長音），長度之後用時間量。 */
function collapse(phones){
  const out=[];let q=false;
  for(const t of phones){
    if(t.p==="Q"){q=true;continue;}
    const last=out[out.length-1];
    if(last&&!q&&isV(t.p)&&last.p===t.p){last.long=true;continue;}
    out.push({p:t.p,opt:!!t.opt,q,long:false});q=false;
  }
  return out;
}
function align(L,phones){
  const toks=collapse(phones),K=toks.length,{F,S,ls}=L,NS=2*K+1,NEG=-1e9;
  const E=toks.map(tk=>{
    const st=setOf(tk.p),e=new Float32Array(F);
    for(let t=0;t<F;t++){if(!st){e[t]=-20;continue;}let s=0;for(const v of st) s+=Math.exp(ls[t*S+v]);e[t]=Math.log(s+1e-12);}
    return e;
  });
  const em=(s,t)=>s%2===0?ls[t*S]:E[(s-1)/2][t];     // 偶數狀態是空白（詞表第 0 個）
  const D=[],B=[];
  for(let t=0;t<F;t++){D.push(new Float32Array(NS).fill(NEG));B.push(new Int16Array(NS).fill(-1));}
  D[0][0]=em(0,0);
  if(K){D[0][1]=em(1,0);if(toks[0].opt&&K>1){D[0][2]=em(2,0);D[0][3]=em(3,0);}}
  for(let t=1;t<F;t++) for(let s=0;s<NS;s++){
    const cand=[s,s-1];
    if(s%2===1){
      const j=(s-1)/2;
      if(j>0&&toks[j-1].p!==toks[j].p) cand.push(s-2);
      if(j>0&&toks[j-1].opt){cand.push(s-3);if(j>1&&toks[j-2].p!==toks[j].p) cand.push(s-4);}   // 跳過可省略的音
    }else{
      const b=s/2;if(b>0&&toks[b-1].opt) cand.push(s-2);
    }
    let best=NEG,bi=-1;
    for(const c of cand){if(c<0)continue;if(D[t-1][c]>best){best=D[t-1][c];bi=c;}}
    if(bi>=0&&best>NEG/2){D[t][s]=best+em(s,t);B[t][s]=bi;}
  }
  const ends=[NS-1,NS-2];
  if(K&&toks[K-1].opt){ends.push(NS-3);if(K>1) ends.push(NS-4);}
  let s=-1,best=NEG;
  for(const e of ends) if(e>=0&&D[F-1][e]>best){best=D[F-1][e];s=e;}
  const on=new Array(K).fill(-1),gop=new Array(K).fill(null);
  if(s>=0) for(let t=F-1;t>=0;t--){
    if(s%2===1){const j=(s-1)/2;on[j]=t;if(gop[j]==null||E[j][t]>gop[j]) gop[j]=E[j][t];}   // gop：這個音在它那幾格裡最高的 log 機率
    const p=B[t][s];if(p<0)break;s=p;
  }
  return {logp:best,F,toks,on,gop};
}

/* ---------- 長度：以「拍」為單位切段，和示範比 ---------- */
function moras(al){
  const tk=al.toks,K=tk.length,st=[];
  for(let k=0;k<K;k++){
    if(al.on[k]<0)continue;
    if(!isV(tk[k].p)||k===0||isV(tk[k-1].p)||tk[k-1].p==="N"||al.on[k-1]<0) st.push(k);   // 每一拍的起點：子音，或前面沒有子音的母音
  }
  const iv=[];
  for(let i=0;i+1<st.length;i++){
    const a=st[i],b=st[i+1];let feat="";
    for(let k=a;k<=b;k++){if(k>a&&tk[k].q) feat="促音";if(k<b&&tk[k].long) feat=feat||"長音";}
    iv.push({a,b,d:al.on[b]-al.on[a],feat,seg:tk.slice(a,b).map(t=>t.p+(t.long?t.p:"")).join("")+(tk[b].q?"っ":"")});
  }
  return {iv,span:st.length?al.on[st[st.length-1]]-al.on[st[0]]:0};
}
const MIN_REST=8;        // 其餘部分至少要這麼多格才拿來校正語速（8 格 = 160 毫秒）
const SHORT_TH=0.8;      // 有長音／促音的那一拍，校正後不到示範的 80% → 太短
const LONG_TH=1.3;       // 測試用：已知位置上超過 130% → 太長
const LONG_HINT=1.5;     // 實際使用：任何一拍超過 150% 才提示（每拍都檢查，門檻要高才不會誤報）
function compareLength(alL,alR){
  const a=moras(alL),b=moras(alR),m=new Map(b.iv.map(x=>[x.a+"-"+x.b,x])),out=[];
  for(const x of a.iv){
    const y=m.get(x.a+"-"+x.b);if(!y||y.d<=0)continue;
    const rl=a.span-x.d,rr=b.span-y.d,norm=rl>=MIN_REST&&rr>=MIN_REST;
    out.push({a:x.a,b:x.b,seg:x.seg,feat:x.feat,dL:x.d,dR:y.d,norm,abs:x.d/y.d,ratio:norm?(x.d/rl)/(y.d/rr):x.d/y.d});
  }
  return out;
}

/* ---------- 清濁音：換成對應的音，比機率 ---------- */
const VP={k:["g"],g:["k"],t:["d"],d:["t"],p:["b"],b:["p"],s:["z"],z:["s"],sh:["j"],ch:["j"],j:["sh","ch"],ky:["gy"],gy:["ky"]};
const KANA_ROW={k:"か行",g:"が行",t:"た行",d:"だ行",p:"ぱ行",b:"ば行",s:"さ行",z:"ざ行",sh:"し",ch:"ち",j:"じ",ky:"きゃ行",gy:"ぎゃ行"};
function voicing(L,phones,base){
  const out=[];
  phones.forEach((t,i)=>{
    const alts=VP[t.p];if(!alts)return;
    let bestAlt=null,bestLp=-1e9;
    for(const a of alts){const alt=phones.map((x,k)=>k===i?{p:a,opt:x.opt}:x),lp=align(L,alt).logp;if(lp>bestLp){bestLp=lp;bestAlt=a;}}
    out.push({i,p:t.p,alt:bestAlt,margin:base.logp-bestLp});
  });
  return out;
}

/* ---------- 一段錄音的完整分析 ----------
   真模型測過之後的結論（合成語音，13 個小段 × 3 種語速）：
   - 長音／促音的長度比對可用：唸錯版抓到 17/18，正確版誤報 1/12。
   - 「每個音有沒有唸出來」（該音的最高機率）不可用：正確版有 19/26 被判成有音沒唸出來。
   - 清濁音不可用：正確版有 7/26 被誤報。
   所以回饋只給長度；每個音的機率和清濁音的差距照樣算出來放在結果裡，供之後分析，不拿來提示學習者。
   長度規則：實際長度（abs）和校正語速後的長度（ratio）都低於門檻才算太短；段落太短無法校正時只看實際長度。 */
const WEAK_P=0.1;        // 只用來在表格上標色，不產生回饋
const ABS_TH=0.9,NORM_TH=0.9;
function analyse(L,kana,R){
  const tg=kanaToPhones(kana),al=align(L,tg),fb=[];
  const phones=al.toks.map((t,k)=>({p:t.p+(t.long?t.p:""),q:t.q,opt:t.opt,on:al.on[k],post:al.gop[k]==null?0:+Math.exp(al.gop[k]).toFixed(3)}));
  const voice=voicing(L,tg,al),length=R?compareLength(al,align(R,tg)):[];
  for(const g of length){
    if(!g.feat)continue;
    const short=g.norm?(g.abs<ABS_TH&&g.ratio<NORM_TH):g.abs<SHORT_TH;
    if(short) fb.push({cat:"short",msg:`${g.feat}太短：「${g.seg}」這一拍的長度是示範的 ${Math.round(g.abs*100)}%${g.norm?`（校正語速後 ${Math.round(g.ratio*100)}%）`:""}`});
  }
  return {target:tg.map(t=>t.opt?"("+t.p+")":t.p).join(" "),greedy:greedy(L).join(" "),logpPerFrame:+(al.logp/L.F).toFixed(3),phones,
    voice:voice.map(v=>({p:v.p,alt:v.alt,margin:+v.margin.toFixed(2)})),length:length.map(g=>({...g,abs:+g.abs.toFixed(2),ratio:+g.ratio.toFixed(2)})),fb};
}

function resample(pcm,f){   // f>1 變慢且音調變低，f<1 變快且音調變高（拿來模擬不同語速的人）
  const n=Math.floor(pcm.length*f),o=new Float32Array(n);
  for(let i=0;i<n;i++){const x=i/f,a=Math.floor(x),b=Math.min(pcm.length-1,a+1);o[i]=pcm[a]*(1-(x-a))+pcm[b]*(x-a);}
  return o;
}

/* ---------- 自我測試（合成語音） ----------
   lsOf(clip, speed) 要回傳該段聲音的 log 機率（speed=1 是原速）。 */
const SPEEDS=[1.2,0.85],SPEED_PAIRS=["s0","s1","s3","s5","long0","long1"];
async function selfTest(clips,lsOf,onStep){
  const byPair={};for(const c of clips)(byPair[c.pair]=byPair[c.pair]||[])[c.side]=c;
  const truth=c=>kanaToPhones(c.said||c.kana),name=c=>c.said||c.kana;
  const rep={voice:{n:0,ok:0,selfFlag:0,selfN:0,rows:[]},length:{n:0,shortHit:0,longHit:0,norm:0,rows:[]},speed:{n:0,okShortFalse:0,shortHit:0,okLongFalse:0,longHit:0,rows:[]}};
  const pairs=Object.values(byPair);let step=0;
  const total=clips.length+SPEED_PAIRS.length*2*SPEEDS.length;
  const get=async(c,sp)=>{const L=await lsOf(c,sp||1);if(onStep) await onStep(++step,total);return L;};
  // 找出一組裡面「有長音／促音」的那一拍在第幾段
  const featIdx=(full,lack)=>{const a=collapse(truth(full)),b=collapse(truth(lack));for(let k=0;k<a.length;k++) if(a[k].q!==b[k].q||a[k].long!==b[k].long) return k;return -1;};
  const segAt=(cmp,alFull,k)=>{   // 包含第 k 個音素（促音）或以它為長音的那一拍
    const mv=moras(alFull).iv;const hit=mv.find(x=>(alFull.toks[k].q&&x.b===k)||(alFull.toks[k].long&&x.a<=k&&k<x.b));
    return hit?cmp.find(g=>g.a===hit.a&&g.b===hit.b)||null:null;
  };
  for(const pr of pairs){
    const LA=await get(pr[0]),LB=await get(pr[1]);
    if(pr[0].cat==="voice"){
      [[pr[0],LA],[pr[1],LB]].forEach(([c,L])=>{
        const a=align(L,truth(pr[0])).logp,b=align(L,truth(pr[1])).logp,pick=a>b?0:1;
        const self=voicing(L,truth(c),align(L,truth(c))).filter(v=>v.margin<0).length;
        rep.voice.n++;if(pick===c.side) rep.voice.ok++;rep.voice.selfN++;if(self) rep.voice.selfFlag++;
        rep.voice.rows.push({said:name(c),diff:+(a-b).toFixed(2),correct:pick===c.side,selfFlags:self});
      });
    }else{
      const full=pr[0].type==="word"?pr[1]:pr[0],lack=pr[0].type==="word"?pr[0]:pr[1],LF=full===pr[0]?LA:LB,LK=full===pr[0]?LB:LA;
      const tg=truth(full),alF=align(LF,tg),alK=align(LK,tg),k=featIdx(full,lack);
      const s=segAt(compareLength(alK,alF),alF,k),l=segAt(compareLength(alF,alK),alF,k);
      rep.length.n++;if(s&&s.norm) rep.length.norm++;
      if(s&&s.ratio<SHORT_TH) rep.length.shortHit++;if(l&&l.ratio>LONG_TH) rep.length.longHit++;
      rep.length.rows.push({target:name(full),said:name(lack),seg:s?s.seg:null,short:s?+s.ratio.toFixed(2):null,long:l?+l.ratio.toFixed(2):null,norm:!!(s&&s.norm),frames:s?[s.dL,s.dR]:null});
    }
  }
  // 語速測試：同一段聲音放慢／加快後當成「學習者」，示範維持原速
  for(const pid of SPEED_PAIRS){
    const pr=byPair[pid];if(!pr)continue;
    const full=pr[0].type==="word"?pr[1]:pr[0],lack=pr[0].type==="word"?pr[0]:pr[1];
    const tg=truth(full),k=featIdx(full,lack),alRF=align(await lsOf(full,1),tg),alRK=align(await lsOf(lack,1),tg);
    for(const sp of SPEEDS){
      const alLF=align(await get(full,sp),tg),alLK=align(await get(lack,sp),tg);
      const okS=segAt(compareLength(alLF,alRF),alRF,k),badS=segAt(compareLength(alLK,alRF),alRF,k);   // 目標有長度：唸對／唸短
      const okL=segAt(compareLength(alLK,alRK),alRF,k),badL=segAt(compareLength(alLF,alRK),alRF,k);   // 目標沒長度：唸對／唸長
      rep.speed.n++;
      if(okS&&okS.ratio<SHORT_TH) rep.speed.okShortFalse++;if(badS&&badS.ratio<SHORT_TH) rep.speed.shortHit++;
      if(okL&&okL.ratio>LONG_TH) rep.speed.okLongFalse++;if(badL&&badL.ratio>LONG_TH) rep.speed.longHit++;
      const f=x=>x?+x.ratio.toFixed(2):null;
      rep.speed.rows.push({pair:pid,speed:sp,target:name(full),okWhenTargetLong:f(okS),shortWhenTargetLong:f(badS),okWhenTargetShort:f(okL),longWhenTargetShort:f(badL)});
    }
  }
  return rep;
}

window.SP={kanaToPhones,normToken,init,setOf,logSoftmax,greedy,collapse,align,moras,compareLength,voicing,analyse,resample,selfTest,SHORT_TH,LONG_TH,LONG_HINT,WEAK_P,ABS_TH,NORM_TH};
})();
