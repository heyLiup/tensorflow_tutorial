var configs = {
  domains: ["www.shujuheiban.com"],
  contentUrlRegexes: [/http:\/\/www\.shujuheiban\.com\/.*/], //内容页url正则
  fields: [
    {
      // 抽取项
      name: "code",
      required: true,
      primaryKey : true,
      selector : /.*/,
      selectorType : SelectorType.Regex,
      sourceType : SourceType.UrlContext
    },
    {
      // 抽取项
      name: "content",
      selectorType : SelectorType.Regex,
      selector : /.*/
    }
  ]
};
configs.initCrawl = function(site){
  //下面回调里面的方法会以异步的方式执行
  //外部的变量和方法想要在里面调用，都需要通过第二个参数传递进去
  site.async(function(params){//这里的params接收async函数的第二个参数
    var baseUrl = "http://www.shujuheiban.com/tutorial/lession-1/captcha-generator.php?label=";
    var pickup = "BCEFGHJKMPQRTVWXY2346789";
    var site = params[0];
    var pos = 0;
    var opt = {
      base64 : true
    };
    for(var i =0;i< 100000;i++ ){
      var captcha = "";
      for(var j=0; j<4; j++){
          pos = Math.round(Math.random() * (pickup.length-1));
          captcha += pickup.charAt(pos);
      }
      var url = baseUrl + captcha;
      opt.contextData = captcha; 
      site.addScanUrl(url,opt);
    }
  }, [site]);
};
configs.afterDownloadPage = function (page, site) {
  page.raw = page.raw.replace(/\+/g,"-").replace(/\//g,"_").replace(/\s+/g,"")
  return page;
};
var crawler = new Crawler(configs);
crawler.start();